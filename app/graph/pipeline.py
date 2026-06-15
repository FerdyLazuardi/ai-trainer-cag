"""
Optimized Agentic RAG pipeline - Retrieve-then-Generate pattern.

Architecture change vs prior ReAct pattern:
  BEFORE: classifier → agent(LLM decides tool) → ToolNode → agent(LLM answers)  = 3 LLM calls for KNOWLEDGE
  AFTER:  classifier → rag_node(pure retrieval) → generate_node(LLM answers)    = 2 LLM calls for KNOWLEDGE

Savings: ~700 tokens per KNOWLEDGE query (the first "decide to call tool" agent call is eliminated).
"""
import asyncio
from functools import lru_cache
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from loguru import logger

from app.config.settings import get_settings
from app.graph.state import RAGState
from app.llm.client import get_chat_llm, get_generate_llm
from app.llm.prompts import PERSONA, OUTPUT_CONTRACT
from app.utils.token_counter import truncate_to_tokens

_settings = get_settings()
_MOODLE_BASE = _settings.moodle_api_url.rstrip("/")


# ─── System Prompts ──────────────────────────────────────────────────────────

CONVERSATIONAL_PROMPT = f"""<role>
{PERSONA}
</role>

{OUTPUT_CONTRACT}

<how_to_talk>
Talk like a senior L&D trainer mentoring a colleague — warm, human, methodical. Respond to what the user ACTUALLY said; if they comment on the conversation itself ("kok gini", "yang bener dong"), acknowledge naturally and move on. A light teaching touch (why it matters, how it ties to <user_context>) is welcome, but never lecture or pad a quick question into a lesson. Mirror the user's language (ID/EN) and tone: casual stays casual, formal stays formal, "aku/kamu" in ID. If <user_preferences> sets a `preferred_tone`, follow it.
</how_to_talk>

<length>
Default: SHORT — 2-4 sentences, warm and direct. Don't pad, don't dump every fact.
Go LONGER and more structured ONLY when the user clearly asks to be taught or wants detail ("jelasin lengkap", "ajarin aku", "gimana caranya step by step", "rinci dong"): then give a fuller answer, numbered steps if it's a procedure, **bold** the key action per step. Use bullets only when genuinely listing items. For a plain definition, answer in prose.
EXCEPTION — the SHORT default does NOT apply when the user asks about a set/category that <context> lists (see the LIST rule in <grounding>): there, completeness wins over brevity — list every item in the context, even if that runs past 4 sentences. A complete bulleted list is the correct SHORT answer for that case.
</length>

<grounding>
First check RELEVANCE: <context> being present does NOT mean it answers this turn. Re-read what the user actually said. If the context genuinely answers their question, base your answer on it. If it does NOT — e.g. the user made a meta-comment about the conversation ("kok ga nyambung", "yang bener dong"), greeted you, or said something the chunks don't actually address — then IGNORE the context entirely and just respond to the user naturally. NEVER pull in a topic from <context> that the user didn't ask about ("Kamu bertanya tentang X..." when they didn't) — that's the worst failure here.
When the context IS relevant: copy Amartha's product, principle, role, and policy names EXACTLY as written in <context> — never swap in a similar-sounding term from general knowledge (e.g. keep Amartha's "Mechanism of Complaints Resolution", don't rename it to the generic CGAP "Grievance Redress"). Do NOT invent Amartha facts (numbers, policies, lists) that aren't in <context>.
When the user asks about a SET or CATEGORY of items — whether phrased explicitly ("produk apa aja", "8 prinsip", "sebutkan semua") OR softly ("produk Amartha", "jenis-jenis X") — answer completely from your FIRST reply; do NOT tease a partial and wait to be pushed. Find the SUMMARY list in <context> (a recap/overview that enumerates the set as one-line bullets) — that summary is the AUTHORITATIVE membership list. Reproduce ALL items from it, ONLY those, with exact names verbatim. Do NOT add items just because their chunk was retrieved (a support service or business-model section is NOT a member). If no summary exists, gather from per-item sections; if some items are still missing, give what's there and say the rest isn't in your materials.
</grounding>

<no_context>
If <context> is absent or doesn't actually answer a factual Amartha question, say so honestly and briefly ("Aku belum nemu info soal itu di materiku — coba pakai kata kunci lain ya") — don't stitch unrelated facts into a fake answer. If the user is clearly off-topic (weather, math, other companies), gently steer back to what you can help with (Amarthapedia materials). If they ask who you are, introduce yourself in one line. If they're just venting or chatting, be human about it — no KB facts forced in.
CRITICAL — acronyms/terms with NO context: if the user asks about an acronym, abbreviation, product, or term ("MBG itu apa", "apa itu XYZ", "kapan ABC cair") and <context> does NOT define it, you MUST say you don't have it — NEVER guess or invent an expansion, definition, or process (do NOT turn "MBG" into a plausible-sounding "Mitra Bisnis Gold"). Inventing a confident expansion for an unknown acronym is the single worst failure here. A real Amartha term would have surfaced in <context>; if it didn't, treat it as unknown and ask the user to clarify or rephrase.
</no_context>"""


# Lite prompt for chit-chat intents (GREETING, AMBIGUOUS, OFF_SCOPE) — no KB
# lookup, no grounding rules, no anti-halu detail. Just a warm persona + brief
# response + steer-back-to-Amarthapedia. Saves ~900 tok per turn vs the full
# CONVERSATIONAL_PROMPT. Cache-eligible: byte-stable, called for ~30% of
# traffic (chit-chat). Routing stays regex (no LLM classifier).
CHIT_CHAT_PROMPT = f"""<role>
{PERSONA}
</role>

{OUTPUT_CONTRACT}

<how_to_talk>
Respond briefly and warmly to the user — like a friendly Amartha colleague, not a search engine. Greet naturally if they greeted; acknowledge the actual content of their message; offer to help with Amarthapedia materials if relevant. If the input is unclear, ask one short clarifying question. If it's off-topic (weather, math, other companies, personal stuff), gently steer back to what you can help with. Keep the reply to 1-3 sentences unless they clearly want more. Mirror their language (ID/EN) and tone.
</how_to_talk>"""


# Socratic coaching prompt — used ONLY when the user opted into Coaching mode
# (ChatRequest.coaching_mode → intent=COACHING). Same persona + anti-leak
# contract as the conversational prompt; the difference is the teaching stance:
# for diagnostic/reasoning questions Ava opens with ONE grounded guiding
# question and keeps each turn LIGHT (minimal validation), holding the full
# confirmation + grounded teaching until the wrap-up. Pure factual lookups are
# still answered directly (asking someone to "guess" an interest rate is absurd).
SOCRATIC_PROMPT = f"""<role>
{PERSONA}
</role>

{OUTPUT_CONTRACT}

<mode>
You are in COACHING mode: the user switched on a "Coaching" toggle because they want to be COACHED through a problem and arrive at the answer themselves, not just handed a quick answer. You are still the same senior L&D trainer — now using the Socratic method. Match the user's language (ID/EN) and use "aku/kamu".
CRITICAL: Output ONLY your final reply to the user. NEVER narrate your own thinking, plans, or decisions (no "The user is...", "I should...", "Sebelum menjawab aku akan..."). NEVER write any tag like <wrap_up> or <mode>. If you catch yourself describing what to do, stop and just do it.

COACHING CONDUCT: never open with apology ("maaf", "kurang pas"), purpose statement ("tujuanku adalah", "aku di sini untuk"), or asking what went wrong. Never close with a generic re-offer ("ada lagi yg bisa kubantu", "feel free to ask"). Vary your opening beat across turns.

FRUSTRATION OVERRIDE: user signals frustration/urgency/critique — "kok gini/gitu", "hah knapa", "yang bener", "capek/cape", "lelah", "buru-buru/cepet", "ga ngerti/ga paham", "bingung/pusing", "nyerah/males/ga mau", or "responnya gini"/"gimana nih"/"salah" — DROP Socratic. State the grounded answer from <retrieved_context> in full, then end with ONE concrete actionable step the user can do in the next 5 minutes. NO "?" anywhere in your reply. If KB doesn't cover it, say so honestly in 1-2 sentences and stop.
</mode>

<scope>
Coaching is for the user's WORK and LEARNING at Amartha: Amarthapedia materials (Client Protection, Anti-Harassment policy, products, BMDP, etc.) and on-the-job challenges (collections, mitra, targets, portfolio quality). It is NOT a personal-life or relationship counseling service.
If the user brings a personal/emotional/relationship matter (breakups, dating a coworker, family, mental health): respond briefly and humanely in 1-2 sentences, do NOT play therapist, do NOT pull KB material to manufacture relevance, and gently steer back to how you CAN help ("Aku di sini buat bantu soal kerjaan dan materi Amarthapedia ya. Ada yang bisa aku bantu di situ?"). If it involves harassment or safety at work, point them to People Care (WhatsApp Satgas PPKS / peoplecare@amartha.com) instead of advising. NEVER assert a topic the user didn't raise (e.g. don't bring up "power relation/consent" unless they asked about it).
</scope>

<when_to_ask_vs_answer>
For a DIAGNOSTIC / REASONING question about the user's own work ("kok mitra aku susah ditagih", "kenapa target ga kecapai", "gimana caranya aku ningkatin repayment"): open with ONE short guiding question that invites them to reason first.
For a PURE FACTUAL LOOKUP (a definition, number, name, policy, or list — "berapa bunga Modal", "apa itu Client Protection", "produk apa aja"): answer DIRECTLY and completely. Do NOT ask them to guess a fact. When unsure, answer directly — a needless quiz is worse than a direct answer.
RE-ASKED TOPIC (important): if the conversation history already contains a full answer to THIS question and the user is now here in Coaching mode, they just opted in to be coached through it — do NOT repeat the previous answer verbatim. Open fresh with ONE guiding question that builds on what they asked, drawing them to reason about it. Make the opener feel natural and tied to THEIR exact wording/keluhan (e.g. they asked "gimana caranya dapetin mitra biar tembus target" → "Oke, kita ulik bareng ya. Menurut kamu, dari mitra yang udah ada vs cari mitra baru, mana yang paling cepat ngangkat pencapaian kamu? Coba tebak dulu — nanti aku konfirm."). NEVER open with a generic "apa yang bikin kamu bingung?" — always anchor to the question they actually asked.
</when_to_ask_vs_answer>

<how_to_ask>
When you do ask a guiding question:
- Ask exactly ONE question, short and concrete — never a list of questions, never a wall of text before it.
- The question must be GROUNDED in <retrieved_context>: hint toward what the materials actually say, don't fish for something not in the KB. You are nudging them toward the real answer, not testing trivia.
- Match the INVITATION verb to what you're asking for. Only say "coba tebak" when there's a factual right answer to guess. If you're asking them to RECALL or SHARE an experience/opinion, invite with "coba ceritakan", "coba inget-inget", or "coba jawab" — NEVER "coba tebak" for something that isn't a guessable fact.
- Always give a light exit so they never feel trapped: e.g. "...atau kalau mau langsung aku jelasin, bilang aja." Vary the wording naturally; don't repeat the exact same exit phrase every turn.
- Example: user asks "kok mitra aku susah ditagih?" → "Sebelum aku jelasin — menurut kamu, apa yang biasanya bikin nasabah mulai susah ditagih? Coba jawab dulu, nanti aku tambahin. (Atau kalau mau langsung, bilang aja.)"
</how_to_ask>

<during_the_loop>
This mode is LIGHT on every intermediate turn. When the user responds to your guiding question with a guess, a partial idea, or shares a real experience:
- Validate MINIMALLY — one short, natural beat that shows you heard them, then move on. A few words is enough ("oke", "noted", "masuk akal", "boleh juga"). Do NOT reflect-back-in-full, do NOT grade right/wrong, do NOT deliver the teaching point yet. The full confirmation and the grounded explanation are deliberately HELD for the wrap-up (see <wrap_up>).
- VERIFY before treating "ga tau" / "bingung" / "ga ngerti" as a stop signal. Three possible user states: (a) tried-and-failed — they've made multiple guesses or partial answers already → wrap up; (b) coy or testing — first or second response, may benefit from a different angle → rephrase the question with a new entry point (a concrete example, an analogy, a hint about WHERE in the KB the answer lives); (c) genuinely stuck on a question they have no lived way to answer (e.g. "what does the partner think?") → rephrase OR wrap up. Read the prior turns — number of attempts, depth of engagement, energy — to judge. A real senior trainer reads the room; this prompt expects you to do the same.
- If you've already asked 3+ guiding questions on the same facet with no progress, wrap up — they're stuck, not coy.
- Then advance with ONE NEW guiding question that goes to the NEXT facet or a level deeper — keep the Socratic thread moving. Light multi-round is the intent: several thin turns, each just nudging forward, NOT a full mini-lesson per turn.
- The new question must be GENUINELY NEW — a different angle, a next step, an application to their case. NEVER re-ask the same question or a trivial reword.
- If they SHARED A REAL EXPERIENCE (not a guess at a fact), acknowledge it as a colleague ("makasih udah cerita") — never grade a lived story with "tepat sekali!". Still keep it short and keep moving.
- Match the invitation verb to the ask: "coba tebak" only for a guessable fact; "coba ceritakan / inget-inget / jawab" for recall or opinion.
- Always pair the next question with a light exit so they never feel trapped ("...atau kalau mau aku langsung rangkum semuanya, bilang aja"). Vary the wording.
- VARY your validation beat across turns. NEVER start two consecutive replies with the same word. Pool: "oke" / "noted" / "masuk akal" / "boleh juga" / "hmm" / "oh gitu" / "fair" / "okay" / "sip". Pick by the user's energy (casual → casual, formal → formal).
</during_the_loop>

<wrap_up>
End the coaching loop and deliver the PAYOFF based on CONTEXT, not just phrase-matching. A senior trainer reads the room and decides; so do you. The user's "ga tau" doesn't always mean "stop" — it might mean "give me a different angle" or "I'm tired of this question" or "I really don't know". Verify per the rule in <during_the_loop> above.

HARD wrap-up triggers (no more Socratic questions, deliver the payoff now):
- The user explicitly asks for the answer: "langsung aja", "kasih tau dong", "bilang aja", "rangkum dong", "udah cukup", "udah".
- The user expresses clear disinterest in continuing: "ga males nebak", "ga mau mikir", "ga usah nebak", "stop nebak", OR "ga tau" / "bingung" repeated after you've already tried multiple angles.
- The user has reasoned their way to (or near) the answer.
- The user signals frustration/urgency (see FRUSTRATION OVERRIDE in <mode>).
- You've walked them through the key facets and there's nothing genuinely new left to probe.
- You've already asked 3+ guiding questions on the same facet with no progress.

SOFT signals (use your judgment, one more probe might still help):
- A first or second "ga tau" / "bingung" / "ga ngerti" / "bingung nih" — verify state per <during_the_loop> rule. If they may benefit from a different angle, rephrase. If they look stuck, wrap up.
- A "hmm" / "..." — they're thinking. Give space, ask one more concrete question or a hint.
At the wrap-up, do the full work you held back during the loop:
- CONFIRM their thinking: tie together what they said across the turns and tell them what was on-point and what needs correcting ("dari yang kamu jawab tadi, soal X kamu udah tepat; yang Y sebenarnya begini...").
- TEACH the grounded answer in full from <retrieved_context> — numbered steps for a procedure, bullets for a list, prose for an explanation. This is the payoff; don't withhold it now. Keep it GROUNDED and COMPLETE but never PADDED — cut filler, don't repeat what the user already showed they understood.
- Close with ONE specific actionable step for THEIR case (e.g. "coba cek angsuran minggu ini, kalau udah >30% income berarti hampir kena batas Maximum Outstanding"). NEVER end with "ada lagi yang bisa kubantu" / "ada yang mau ditanyakan lagi" / "feel free to ask" / "ada lagi yg mau didiskusikan" — those are generic chatbot closers, NOT senior-trainer closers.
The goal is for the user to ARRIVE at understanding, NOT to make them admit they're ignorant. Warm senior-to-junior coaching, never an adversarial gotcha. (Frustration handling — when to drop Socratic entirely — lives in <hard_rules> above; do not duplicate here.)
</wrap_up>

<grounding>
Everything you assert — and every guiding question's premise — must be grounded in <retrieved_context>. Copy Amartha's product, principle, role, and policy names EXACTLY as written; never invent facts, numbers, or policies. If the context doesn't actually cover what they asked, say so honestly ("Aku belum nemu ini di materiku") instead of inventing a question or an answer around it. The teaching tone never overrides faithfulness.
</grounding>

<length>
Guiding question (loop turns): 1-3 sentences, light and inviting — keep each intermediate turn short (a brief validation beat + one question). The wrap-up teaching answer: as long as needed to be complete and grounded — numbered steps for a procedure, bullets for a list, prose for an explanation. Warm but never padded.
</length>"""


# ─── Nodes ───────────────────────────────────────────────────────────────────

# Strips leaked instruction blocks from the LLM response. Some models
# (Gemini Flash Lite especially) occasionally echo the literal contents of
# <retrieved_context> / <user_history> / etc. as part of their output —
# leading to giant <h1>-rendered context dumps in the UI. We catch that
# server-side as a defensive net even after prompt-level guards.
_LEAK_BLOCK_RE = re.compile(
    r"<(retrieved_context|user_history|previous_context|user_preferences|user_context|response_shape|conversation_signals|capabilities|mode|output_contract|role|rules|how_to_talk|length|grounding|no_context|when_to_ask_vs_answer|how_to_ask|during_the_loop|wrap_up|scope|available_topics)>"
    r".*?"
    r"</\1>\s*",
    re.DOTALL | re.IGNORECASE,
)
_LEAK_OPEN_TAG_RE = re.compile(
    r"</?(retrieved_context|user_history|previous_context|user_preferences|user_context|response_shape|conversation_signals|capabilities|mode|output_contract|role|rules|how_to_talk|length|grounding|no_context|when_to_ask_vs_answer|how_to_ask|during_the_loop|wrap_up|scope|available_topics)>",
    re.IGNORECASE,
)
# Citation header from context formatter — "[N] Course: <name> (ID:<id>)".
# Distinctive pattern; never appears in legitimate prose.
_LEAK_CITATION_HEAD_RE = re.compile(
    r"^\s*\[\d+\]\s*Course:\s*[^\n]*?\(ID:[^)]*\)",
    re.MULTILINE | re.IGNORECASE,
)
# ATX markdown headings — "# Foo", "## Bar". Stripping these from chunk text
# before sending to the LLM prevents the giant-font rendering disaster if
# the LLM later echoes chunk content verbatim.
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+", re.MULTILINE)
# Inline source citations like "[[1]]" or "[[1]][[2]]" that the LLM
# sometimes emits from the persona's old example format. Sources are
# rendered separately in the UI — never inline in the user-facing reply.
_INLINE_CITE_RE = re.compile(r"\[\[\d+\]\]")
# Layer-4 leak: lines that look like LITERAL prompt directives (the LLM
# drifts into reciting its conditioning when it has no good answer). They
# start with rule-list words ("Default:", "Go LONGER", "EXCEPTION", "NEVER
# ...", "Open with", "End with", "Talk like", etc.) and are NOT natural
# prose. This catches the case where the LLM echoes block CONTENTS without
# the wrapping tags (Layers 1-3 only catch tagged leaks).
_DIRECTIVE_LINE_RE = re.compile(
    r"^[ \t]*(?:"
    r"Default\s*:\s*SHORT|"
    r"Go LONGER and more structured|"
    r"EXCEPTION\s*[—–-]|"
    r"NEVER\s+(?:echo|pull|use|close|start|emit|start|open)|"
    r"ALWAYS\s+(?:open|close|preserve|use|emit|start)|"
    r"Open with the answer|"
    r"End with substance|"
    r"No hedging|"
    r"Use complete sentences|"
    r"Use bullets for lists|"
    r"Mirror the user's language|"
    r"If <context> is absent|"
    r"When the context (?:IS|is) relevant|"
    r"When the user asks about (?:a SET|the set)|"
    r"CRITICAL\s*[—–-]|"
    r"Talk like a senior|"
    r"Answer factual (?:lookups|questions)|"
    r"Format examples \(Indonesian\)|"
    r"STYLE\s*[—–-]|"
    r"MENTOR MINDSET|"
    r"In COACHING mode|"
    r"FRUSTRATION OVERRIDE|"
    r"COACHING CONDUCT|"
    r"First check RELEVANCE|"
    r"When the context IS relevant"
    r")"
    # Eat the rest of the line (often continues with quoted examples / em-dash rules)
    r"[^\n]*",
    re.MULTILINE | re.IGNORECASE,
)

# Meta-conversation recall questions ("udah bahas apa aja", "yang kita bahas",
# "emng itu aja yang kita bahas", "what did we discuss"). The answer is the
# conversation history, NOT the knowledge base — so _pre_processor routes these
# to the no-retrieval path. Without this, the question gets embedded + retrieved,
# random chunks cross the dense floor, and the model describes THOSE as "what we
# discussed" (the fabrication bug). Deliberately biased toward catching meta
# questions (a false positive merely answers from history; a false negative
# brings back the fabrication). A missed phrasing falls through to KNOWLEDGE,
# where the prompt's relevance gate + the wider history window are the backstop.
_META_CONVO_RE = re.compile(
    r"(?:udah|sudah|udh|tadi|barusan|kita|kami)\b[^.?!\n]{0,30}"
    r"(?:bahas|dibahas|ngomong|omongin|diskusi|obrol)"
    r"|(?:yang|apa)\b[^.?!\n]{0,20}(?:di)?(?:bahas|omongin|diskusi)"
    r"|itu aja[^.?!\n]{0,25}(?:bahas|omongin)"
    r"|what (?:did|have|were) we (?:discuss|talk|cover|go over|chat)"
    # Short deictic follow-ups: "which one?" / "the earlier one?" /
    # "give me an example?" — need clarification, not KB retrieval.
    r"|(?:yg|yang)\s+(?:mana|tadi|yg\s+tadi|sebelumnya|sebelum|yg\s+sebelumnya)\b"
    r"|(?:yg|yang)\s+(?:mana|tadi|sebelumnya)\s*[?.!\s]*$"
    r"|(?:bisa|kasih|bs|boleh|blh)\s+(?:kasih|beri|berikan|ada)\s+(?:contoh|contohin)\b"
    r"|(?:kasih|beri|berikan|ada)\s+contoh\b"
    r"|(?:gimana|gmana|gmn|how)\s+(?:caranya|carany|caranya\s+ya)\b"
    r"|(?:terus|trus|lanjut|next)\s+(?:gimana|gmn|apa|apanya)\b",
    re.IGNORECASE,
)

# Pure anaphoric/deictic follow-up markers — a short turn that references the
# PRIOR turn without naming its own topic ("jelasin lagi", "terus gimana",
# "kok gitu", "contohnya?", "yang tadi"). Only these get the prior-user-turn
# prepend for retrieval. Deliberately EXCLUDES bare "kenapa"/"gimana"/"apa",
# which routinely open SELF-CONTAINED questions ("kenapa CP penting", "gimana
# cara cegah fraud") that must retrieve on their own text. Requiring the -nya
# forms (contohnya/detailnya) avoids matching self-contained "kasih contoh X".
_ANAPHORIC_FOLLOWUP_RE = re.compile(
    r"\b(?:lagi|tadi|terus|trus|lanjut(?:in|kan)?|gitu|gini|"
    r"contohnya|maksudnya|selengkapnya|detailnya|"
    r"sebelumnya|berikutnya|selanjutnya)\b"
    r"|\b(?:abis|habis|setelah)\s+itu\b",
    re.IGNORECASE,
)


def _strip_md_headings_for_context(text: str) -> str:
    """Strip ATX markdown headings (#, ##, ###) from chunk text.

    Reason: chunks come from Markdown KB documents, so they contain "# Title"
    lines. If the LLM echoes a chunk verbatim, the frontend renders those
    headings as <h1>/<h2>, producing fonts 2-4x normal body. Stripping the
    leading "#" makes the text plain — even on echo, the UI stays sane.
    Bold/italic/lists are preserved (only headings are visually catastrophic).
    """
    return _MD_HEADING_RE.sub("", text)


def _sanitize_answer(text: str) -> str:
    """Strip any leaked instruction-block content / tags from an LLM reply.

    Layer 1: balanced XML wrappers (when LLM echoes the whole tag block).
    Layer 2: orphan tags (when only one half leaked).
    Layer 3: bare context dump (when LLM dropped XML wrapper but kept the
             "[N] Course: <name> (ID:<id>)" citation headers and chunk text).
             Heuristic: if any citation header is present, assume everything
             before the LAST one is leak. Take the tail after the last header
             and drop the first paragraph (chunk text) — keep the rest as
             the actual answer. Falls back to a generic retry message if
             nothing usable remains.
    """
    if not text:
        return text
    cleaned = _LEAK_BLOCK_RE.sub("", text)
    cleaned = _LEAK_OPEN_TAG_RE.sub("", cleaned)
    cleaned = _INLINE_CITE_RE.sub("", cleaned)
    # Layer 4: strip prompt-directive echoes (untagged prompt content the
    # LLM recites when it has no good answer — e.g. "Default: SHORT — 2-4
    # sentences..." from the <length> block, leaked without its wrapper).
    cleaned = _DIRECTIVE_LINE_RE.sub("", cleaned)
    # Collapse 3+ consecutive blank lines that the stripping may leave behind
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    matches = list(_LEAK_CITATION_HEAD_RE.finditer(cleaned))
    if matches:
        last_end = matches[-1].end()
        tail = cleaned[last_end:].strip()
        # The chunk text right after the last citation header is still leak.
        # Drop the first paragraph; keep whatever follows.
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", tail) if p.strip()]
        if len(paragraphs) >= 2:
            cleaned = "\n\n".join(paragraphs[1:])
        elif paragraphs:
            # Only one paragraph — could be the answer OR could be just the
            # chunk text. Heuristic: if it's longer than 80 chars and doesn't
            # start with a bullet/dash, assume it's the answer.
            only = paragraphs[0]
            if len(only) > 80 and not only.startswith(("-", "*", "•")):
                cleaned = only
            else:
                cleaned = "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."
        else:
            cleaned = "Maaf, ada kendala merangkum jawaban. Coba tanya ulang ya."

    return cleaned.lstrip()


class StreamLeakGuard:
    """Stream-time leak detector. Buffers the first ~80 chars of a generated
    reply so raw-context leaks (chunk citations like "[1] Course: X (ID:N)" or
    bare <retrieved_context> tags) never reach the user mid-stream. The end-of-
    stream `_sanitize_answer` already protects cache/history/eval — this class
    is the user-facing complement so the leak isn't visible in real time.

    Modes:
      - preamble: buffer until PREAMBLE_LIMIT chars accumulated, then decide.
      - passthrough: preamble was clean — yield raw tokens straight through.
      - buffered: leak detected — keep accumulating, sanitize at flush().
    """

    PREAMBLE_LIMIT = 80
    _LEAK_PATTERNS = (
        _LEAK_CITATION_HEAD_RE,
        _LEAK_OPEN_TAG_RE,
        _INLINE_CITE_RE,
        _DIRECTIVE_LINE_RE,
    )

    def __init__(self) -> None:
        self._buffer = ""
        self._mode = "preamble"

    def feed(self, token: str) -> str:
        """Push a streamed token. Returns the safe text to emit (may be "")."""
        if self._mode == "passthrough":
            return token
        self._buffer += token
        if self._mode == "buffered":
            return ""
        if len(self._buffer) < self.PREAMBLE_LIMIT:
            return ""
        if any(p.search(self._buffer) for p in self._LEAK_PATTERNS):
            self._mode = "buffered"
            logger.warning(
                "StreamLeakGuard: leak signature detected in preamble — "
                "switching to buffered/sanitize mode"
            )
            return ""
        out = self._buffer
        self._buffer = ""
        self._mode = "passthrough"
        return out

    def flush(self) -> str:
        """Called at end-of-stream. Returns sanitized trailing text."""
        if self._mode == "buffered":
            cleaned = _sanitize_answer(self._buffer)
            self._buffer = ""
            return cleaned
        out = self._buffer
        self._buffer = ""
        return out

    @property
    def leak_detected(self) -> bool:
        return self._mode == "buffered"


async def _incr_parse_failure_metric() -> None:
    """Fire-and-forget counter for pre-processor JSON parse failures (C2).

    Bucketed by UTC date so the key self-expires (7-day retention) and gives a
    per-day failure rate that ops can scrape with a single SCAN/GET. Never
    raises — a metrics write must never break the request path. A rising count
    here means the pre-processor is failing to classify cleanly often enough to
    fall back to a default intent, i.e. silent quality decay.
    """
    try:
        from datetime import datetime, timezone

        from app.database.redis_client import get_redis_client

        redis = get_redis_client()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        key = f"rag:metrics:preprocess_parse_failure:{day}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 7 * 24 * 3600)
        await pipe.execute()
    except Exception:
        pass


# Strong refs for fire-and-forget background metric tasks scheduled from sync
# routers (otherwise the event loop may GC the task mid-flight).
_BACKGROUND_TASKS: set[asyncio.Task] = set()


async def _incr_sparse_only_passthrough_metric() -> None:
    """Fire-and-forget counter for KNOWLEDGE queries that clear the relevance
    gate on the SPARSE (BM25 lexical) signal ALONE (H9).

    Dense cosine was below its floor but a raw term match rescued the query.
    Bucketed by UTC date (7-day retention); mirrors _incr_parse_failure_metric.
    This is the early-warning signal for the colloquial-ID false-pass risk:
    terse / slangy Indonesian queries that dense embeddings rank off-topic but
    happen to share a BM25 term with the KB. When `rag:metrics:gate_sparse_only_pass`
    climbs, pull those queries and grow the eval sample to confirm the OR-gate
    isn't waving through hallucination-prone misses. Never raises.
    """
    try:
        from datetime import datetime, timezone

        from app.database.redis_client import get_redis_client

        redis = get_redis_client()
        day = datetime.now(timezone.utc).strftime("%Y%m%d")
        key = f"rag:metrics:gate_sparse_only_pass:{day}"
        pipe = redis.pipeline()
        pipe.incr(key)
        pipe.expire(key, 7 * 24 * 3600)
        await pipe.execute()
    except Exception:
        pass


def _emit_sparse_only_passthrough() -> None:
    """Schedule the sparse-only-pass counter from a sync router without blocking.

    `_route_after_rag` is a sync LangGraph router, so we can't await. Schedule
    the Redis incr on the running loop and hold a strong ref so it isn't GC'd
    mid-flight. If no loop is running (a unit test calling the router directly),
    swallow the RuntimeError — the f-string log line above is the fallback
    signal, and a metrics write must never break routing.
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(_incr_sparse_only_passthrough_metric())
        _BACKGROUND_TASKS.add(task)
        task.add_done_callback(_BACKGROUND_TASKS.discard)
    except RuntimeError:
        pass


def _log_cache_usage(response: Any, call_name: str) -> None:
    """Log OpenRouter/Gemini prompt-cache hit info for ONE LLM call.

    The chat path previously logged NOTHING about cache effectiveness, so a
    cache regression (a prompt dropping below the provider's cache-min token
    threshold, or a cache_control breakpoint silently not honored) was
    invisible — only inferrable from the OpenRouter dashboard. This surfaces
    cached-prompt-token counts per call so we can SEE whether the cache is hit.
    Best-effort: never raises.

    LangChain and the raw gateway expose this differently, so check both:
      - usage_metadata.input_token_details.cache_read  (LangChain-normalized)
      - response_metadata.token_usage.prompt_tokens_details.cached_tokens (raw)
    """
    try:
        cached = 0
        prompt = 0
        um = getattr(response, "usage_metadata", None) or {}
        if um:
            prompt = um.get("input_tokens", 0) or 0
            details = um.get("input_token_details") or {}
            cached = details.get("cache_read", 0) or 0
        if not cached or not prompt:
            rm = getattr(response, "response_metadata", None) or {}
            tu = rm.get("token_usage") or {}
            prompt = prompt or (tu.get("prompt_tokens", 0) or 0)
            ptd = tu.get("prompt_tokens_details") or {}
            cached = cached or (ptd.get("cached_tokens", 0) or 0)
        pct = (cached / prompt * 100) if prompt else 0.0
        logger.info(
            "LLM cache usage [{}]: cached={}/{} prompt tok ({:.0f}%)",
            call_name, cached, prompt, pct,
        )
    except Exception as e:
        logger.debug("cache-usage log skipped [{}]: {}", call_name, e)


async def _pre_processor(state: RAGState, config: RunnableConfig):
    """Lightweight pre-step — NO LLM call. Decides retrieval vs no-retrieval.

    Ava is one conversational LLM call (see _generate_node + CONVERSATIONAL_PROMPT).
    This node uses the deterministic regex Tier-1 classifier ONLY to route — it
    never emits a canned reply (that was the old "yang benerlah → identity intro"
    misroute). Three buckets:
      - MALICIOUS (injection/jailbreak) → canned refusal, no retrieval, no LLM.
      - CHIT-CHAT (GREETING / AMBIGUOUS / OFF_SCOPE / TOPIC_LIST): a salutation,
        identity Q, vague filler, off-topic, or "what topics exist" — these need
        NO knowledge-base lookup, so we SKIP retrieval and go straight to the
        conversational generate node with NO <context>. That prevents an
        irrelevant chunk from being dumped into a greeting/vague turn, and lets
        the prompt ask a clarifying question on ambiguous input instead of
        guessing. Cheaper too (no embed + no Qdrant round-trip).
      - KNOWLEDGE (regex returns None — a real question): retrieve, then generate.

    `intent` carries the regex label so chat.py's existing cache/eval gates
    (which already exclude GREETING/AMBIGUOUS/etc.) keep working. `intent_scores`
    stays a vestigial derived dict for the DB/logging schema.
    """
    from app.graph.intent_rules import classify as rule_classify

    messages = state["messages"]
    user_msg = messages[-1].content
    user_msg_str = user_msg if isinstance(user_msg, str) else str(user_msg)

    rule_intent = rule_classify(user_msg_str)

    # ── Injection / jailbreak guard ─────────────────────────────────────────
    if rule_intent == "MALICIOUS":
        logger.info("Pre-processor: injection detected → MALICIOUS")
        return {
            "intent": "MALICIOUS",
            "rewritten_query": user_msg_str,
            "retrieval_query": user_msg_str,
            "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
        }

    # ── Meta-conversation question → answer from HISTORY, never the KB ───────
    # "kita udah bahas apa aja", "tadi ngomongin apa", "what did we discuss" —
    # the answer is the conversation itself, NOT a knowledge-base lookup. If we
    # retrieved, random chunks crossing the dense floor would be described as
    # "what we discussed" (the fabrication bug). Route to the no-retrieval path
    # so generate_node answers purely from the windowed message history.
    if _META_CONVO_RE.search(user_msg_str):
        logger.info("Pre-processor: meta-conversation question → no retrieval (answer from history)")
        return {
            "intent": "AMBIGUOUS",  # no-retrieval bucket; excluded from cache/eval in chat.py
            "rewritten_query": user_msg_str,
            "retrieval_query": user_msg_str,
            "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
        }

    # NOTE: "apa aja di <section>" text-detection was REMOVED — structured
    # navigation (which section, which item) now lives in the UI: a topic-list
    # button opens a section/item picker, and clicking an item sends a normal
    # KNOWLEDGE query ("jelaskan tentang <item>"). Free-text section parsing was
    # fragile (cross-language, content-noun collisions) and is no longer needed.
    # The full topic list ("topik apa aja") still routes via the regex/semantic
    # TOPIC_LIST path below.

    # ── Chit-chat / no-lookup intents → skip retrieval entirely ─────────────
    if rule_intent in ("GREETING", "AMBIGUOUS", "OFF_SCOPE", "TOPIC_LIST"):
        logger.info(f"Pre-processor: {rule_intent} → no retrieval, straight to generate")
        return {
            "intent": rule_intent,
            "rewritten_query": user_msg_str,
            "retrieval_query": user_msg_str,
            "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
        }

    # ── KNOWLEDGE: a real question → retrieve, then generate ────────────────
    # Best-effort follow-up context (no LLM): a terse ANAPHORIC follow-up
    # ("jelasin lagi", "terus gimana", "kenapa gitu") doesn't retrieve well
    # alone, so prepend the most recent prior USER turn to land the search near
    # the active topic.
    #
    # GUARD (fixes topic-panel bug): only prepend when the message is genuinely
    # anaphoric — i.e. it references the prior turn WITHOUT naming its own topic.
    # A self-contained short query ("jelaskan tentang Pelayanan", "apa itu PAR")
    # names its topic and retrieves correctly alone; blending the previous turn
    # into it pulls the WRONG document. The topic panel sends consecutive
    # "jelaskan tentang X" clicks (each ≤40 chars), so the old length-only rule
    # made every click inherit the PREVIOUS topic's chunks → a confident
    # "belum nemu". Requiring a pure deictic marker (gitu/lagi/tadi/terus/…,
    # deliberately NOT "kenapa"/"gimana" which commonly open self-contained
    # questions like "kenapa CP penting") keeps the follow-up benefit while
    # leaving self-contained queries untouched.
    retrieval_query = user_msg_str
    _msg_stripped = user_msg_str.strip()
    if (
        len(_msg_stripped) <= 40
        and len(messages) > 1
        and _ANAPHORIC_FOLLOWUP_RE.search(_msg_stripped)
    ):
        prior_user = next(
            (
                (m.content if isinstance(m.content, str) else str(m.content))
                for m in reversed(messages[:-1])
                if isinstance(m, HumanMessage)
            ),
            None,
        )
        if prior_user:
            retrieval_query = f"{prior_user.strip()} {_msg_stripped}".strip()

    # ── Semantic TOPIC_LIST fallback (regex missed) ─────────────────────────
    # The regex Tier-1 can't catch every typo/paraphrase of "what can I learn?"
    # ("materi yang kamu bisa pelajari", "bisa belajar apa hari ini"). These fell
    # through to KNOWLEDGE → retrieved random chunks → the model listed granular
    # sub-topics instead of the real section list. The embedding centroid
    # recognises them cleanly (>=0.70, best=TOPIC_LIST), while content questions
    # like "produk amartha apa aja" score ~0.58/best=KNOWLEDGE and correctly
    # DON'T match.
    #
    # COST GUARD (pre-gate): the semantic check needs a fresh embed (~390ms), so
    # we must NOT run it on every KNOWLEDGE turn. A topic-list question ALWAYS
    # mentions a learning/topic hint word — so we first do a cheap substring
    # gate. Pure content questions ("berapa bunga modal", "apa itu client
    # protection") have no hint word → skip the embed entirely → zero added
    # latency. Only the rare hint-bearing phrasing the regex missed pays the
    # embed (and _embed_one caches it). Length-bounded so long real questions
    # that happen to contain "materi" ("jelasin materi CP dong panjang lebar...")
    # don't trigger the embed either.
    _low_msg = user_msg_str.lower()
    _TL_HINTS = ("belajar", "pelajar", "dipelajari", "materi", "topik", "tema",
                 "konten", "course", "kursus", "pelatihan", "modul", "pembelajaran")
    _tl_pregate = len(_low_msg) <= 60 and any(h in _low_msg for h in _TL_HINTS)
    if _tl_pregate and not state.get("coaching_mode"):
        try:
            from app.graph.intent_classifier import is_topic_list_semantic
            # Fresh embed inside the check (reusing the route embedding gave
            # inconsistent borderline scores). Cached by _embed_one, and gated
            # above so it only runs on hint-bearing, regex-missed phrasings.
            if await is_topic_list_semantic(user_msg_str):
                logger.info("Pre-processor: semantic TOPIC_LIST fallback → no retrieval")
                return {
                    "intent": "TOPIC_LIST",
                    "rewritten_query": user_msg_str,
                    "retrieval_query": user_msg_str,
                    "intent_scores": {"needs_lookup": 0.0, "needs_reasoning": 0.0, "needs_empathy": 0.0, "needs_safety_escalation": 0.0, "learning_context": 0.0},
                }
        except Exception as exc:
            logger.debug(f"semantic TOPIC_LIST fallback skipped: {exc}")

    # ── Coaching (Socratic) promotion ───────────────────────────────────────
    # When the user has the coaching toggle ON (state.coaching_mode), a real
    # question becomes a COACHING turn instead of KNOWLEDGE. generate_node then
    # uses SOCRATIC_PROMPT — which opens diagnostic/reasoning asks with ONE
    # grounded guiding question, but still answers pure factual lookups directly
    # (that fact-vs-diagnostic split is an LLM judgment in the prompt, not a
    # fragile regex here). Retrieval runs either way: a guiding question must be
    # grounded in the KB, not invented.
    intent = "COACHING" if state.get("coaching_mode") else "KNOWLEDGE"
    logger.info(f"Pre-processor: intent={intent} retrieval='{retrieval_query[:60]}...'")
    return {
        "intent": intent,
        "rewritten_query": user_msg_str,
        "retrieval_query": retrieval_query,
        "intent_scores": {
            "needs_lookup": 1.0,
            "needs_reasoning": 1.0 if intent == "COACHING" else 0.0,
            "needs_empathy": 0.0,
            "needs_safety_escalation": 0.0,
            "learning_context": 0.0,
        },
    }


async def _handle_malicious(state: RAGState, config: RunnableConfig):
    """Canned refusal for jailbreak/prompt-injection (deterministic guard).

    The only canned handler kept after the conversational collapse. _is_injection
    in _pre_processor routes here BEFORE any retrieval/LLM, so an injection attempt
    never reaches the conversational prompt. No LLM.
    """
    from langchain_core.messages import AIMessage
    return {"messages": [AIMessage(content=(
        "Maaf, tugasku khusus untuk membantu seputar materi Amarthapedia dan "
        "kebijakan internal Amartha. Ada yang bisa kubantu seputar itu?"
    ))]}


async def _rag_node(state: RAGState, config: RunnableConfig):
    """
    Pure retrieval node — calls hybrid_search (dense + sparse BM25 fusion)
    without any LLM call. Stores formatted context chunks into
    state['retrieved_context']. This replaces the first ReAct 'agent' call that
    previously just decided to use a tool.
    """
    from app.retrieval.hybrid_retriever import hybrid_search

    # Use the guarded retrieval query if available; it preserves critical
    # anchors that a standalone rewrite may legitimately hide from the UI.
    query_to_search = (
        state.get("retrieval_query")
        or state.get("rewritten_query")
        or state["messages"][-1].content
    )

    # H5 — reuse the embedding the chat route already computed, but ONLY if it
    # was computed from the exact text we're about to search. The route embeds
    # `resolved_query` (for cache/LTM); after rewrite/safety-anchoring the
    # retrieval query often differs, in which case reusing would search the
    # wrong vector — so we fall back to embedding inside hybrid_search.
    precomputed = state.get("query_embedding")
    precomputed_text = state.get("query_embedding_text")
    reuse_embedding = (
        precomputed
        if precomputed and precomputed_text == query_to_search
        else None
    )

    try:
        result = await hybrid_search(
            query=query_to_search,  # type: ignore[arg-type]  # langchain message.content is str at runtime
            top_k=_settings.final_top_k,
            query_embedding=reuse_embedding,
        )
        docs = result.chunks

        chunks = []
        for d in docs:
            m = d.metadata or {}
            chunks.append({
                "text": d.text,
                "course_id": m.get("course_id", ""),
                "course_name": m.get("course_name", d.title),
                "score": round(d.score, 4) if d.score is not None else 0.0,
                "hybrid_score": round(d.hybrid_score, 4) if d.hybrid_score is not None else 0.0,
                "dense_score": round(d.dense_score, 4) if d.dense_score is not None else 0.0,
                "sparse_score": round(d.sparse_score, 4) if d.sparse_score is not None else 0.0,
                "source": d.source or m.get("source", "Unknown"),
                "document_id": d.document_id or m.get("document_id", "Unknown"),
            })

        logger.info(f"RAG node retrieved {len(chunks)} chunks for query: {query_to_search[:60]}")
        # C4: surface pool-level signals (max over full fetch_k pool, pre-slice)
        # for the NOT-FOUND gate; C5: dense_retrieval_ok=False when degraded to
        # sparse-only so the gate doesn't read the missing dense signal as a miss.
        return {
            "retrieved_context": chunks,
            "pool_max_dense": result.pool_max_dense,
            "pool_max_sparse": result.pool_max_sparse,
            "dense_retrieval_ok": result.dense_available,
        }

    except Exception as e:
        logger.error(f"RAG node retrieval failed: {e}")
        # Raise instead of swallowing to allow FastAPI's error handler to return 500
        raise RuntimeError(f"Database error during context retrieval: {e}") from e


def _window_generate_history(messages: list, max_fresh_turns: int, max_ai_chars: int) -> list:
    """Trim the message history fed to generate_node.

    chat.py hands generate_node the current query (always the LAST message)
    preceded by up to `get_or_summarize_history`'s window of completed turns;
    everything older is already folded into the rolling summary
    (<previous_context>). So feeding the full turn list here double-pays:
    the summary covers the old turns AND the raw turns are still attached.

    Two cuts:
      1. Keep only the last `max_fresh_turns` completed turns (= 2*N messages)
         before the current query, then re-append the current query.
      2. Cap each AIMessage's content to `max_ai_chars` — prior AI replies can
         be long, and only their gist (entity names, the topic in play) matters
         for follow-up resolution. User turns are left intact (short + carry the
         actual intent).

    Returns a NEW list with NEW capped AIMessage objects, so state["messages"]
    (consumed downstream for history/cache persistence) is never mutated.
    """
    if not messages:
        return messages
    current = messages[-1]
    prior = messages[:-1]
    if max_fresh_turns > 0 and len(prior) > max_fresh_turns * 2:
        prior = prior[-(max_fresh_turns * 2):]

    windowed: list = []
    for m in prior:
        if isinstance(m, AIMessage):
            content = m.content if isinstance(m.content, str) else str(m.content)
            if max_ai_chars and len(content) > max_ai_chars:
                windowed.append(AIMessage(content=content[:max_ai_chars].rstrip() + "…"))
                continue
        windowed.append(m)
    windowed.append(current)
    return windowed


_COURSE_CACHE_TTL_SECONDS = 600  # 10 minutes
_course_cache: dict[str, Any] = {"courses": [], "expires_at": 0.0}
_course_cache_lock: asyncio.Lock | None = None


def _get_course_cache_lock() -> asyncio.Lock:
    """Lazy-init the cache lock (must be created inside a running event loop)."""
    global _course_cache_lock
    if _course_cache_lock is None:
        _course_cache_lock = asyncio.Lock()
    return _course_cache_lock


async def _load_course_names() -> list[str]:
    """Distinct TOPIC labels from the documents table, TTL-cached (10min).

    The ground-truth list of topics Ava actually has, injected into the generate
    prompt on a TOPIC_LIST turn so "ada materi apa aja" is answered from real
    data instead of fabricated. Single-flight via asyncio.Lock so concurrent
    requests don't stampede Postgres on cache expiry.

    Topic label = the Moodle SECTION name when present, else the per-file
    `course_name` (backward-compat for docs ingested before section_name
    existed). COALESCE(NULLIF(section_name,''), course_name) means several files
    in one Moodle section (e.g. modal.md + celengan.md under "Product Amartha")
    collapse to ONE topic entry instead of spamming one row per file — the
    manual Moodle section structure becomes the topic taxonomy.
    """
    import time as _time

    now = _time.time()
    if now < _course_cache["expires_at"] and _course_cache["courses"]:
        return _course_cache["courses"]

    lock = _get_course_cache_lock()
    async with lock:
        now = _time.time()
        if now < _course_cache["expires_at"] and _course_cache["courses"]:
            return _course_cache["courses"]

        from sqlalchemy import select, distinct
        from sqlalchemy.sql import text as sql_text
        from app.database.postgres import AsyncSessionLocal

        try:
            async with AsyncSessionLocal() as session:
                # Prefer the Moodle section name; fall back to course_name when a
                # doc has no section_name (pre-section_name ingests). The outer
                # filter drops rows where BOTH are empty.
                topic_expr = (
                    "COALESCE(NULLIF(metadata->>'section_name', ''), "
                    "metadata->>'course_name')"
                )
                stmt = (
                    select(distinct(sql_text(topic_expr)).label("topic"))
                    .select_from(sql_text("documents"))
                    .where(sql_text(f"{topic_expr} IS NOT NULL"))
                    .where(sql_text(f"{topic_expr} <> ''"))
                )
                rows = (await session.execute(stmt)).all()
                courses = sorted({r.topic for r in rows if r.topic})
        except Exception as exc:
            logger.warning(f"Topic-name load failed (Postgres): {exc}")
            return []

        _course_cache["courses"] = courses
        _course_cache["expires_at"] = now + _COURSE_CACHE_TTL_SECONDS
        return courses


# ── Section → items map (for "apa aja di <section>" questions) ────────────────
_section_map_cache: dict[str, Any] = {"map": {}, "expires_at": 0.0}
_section_map_lock: asyncio.Lock | None = None


def _get_section_map_lock() -> asyncio.Lock:
    global _section_map_lock
    if _section_map_lock is None:
        _section_map_lock = asyncio.Lock()
    return _section_map_lock


async def _load_section_map() -> dict[str, list[str]]:
    """Map each Moodle SECTION → its item (course_name) list, TTL-cached (10min).

    Ground truth for "apa aja di <section>" / "isi <section>" questions: a section
    like "Business Process" lists its files (Validasi UK, Pelayanan, ...). Pulled
    straight from Postgres so the answer is deterministic, never an LLM guess.
    Only includes docs that actually have a section_name (the per-section grouping
    only makes sense for section-tagged docs).
    """
    import time as _time

    now = _time.time()
    if now < _section_map_cache["expires_at"] and _section_map_cache["map"]:
        return _section_map_cache["map"]

    lock = _get_section_map_lock()
    async with lock:
        now = _time.time()
        if now < _section_map_cache["expires_at"] and _section_map_cache["map"]:
            return _section_map_cache["map"]

        from sqlalchemy.sql import text as sql_text
        from app.database.postgres import AsyncSessionLocal

        section_map: dict[str, list[str]] = {}
        try:
            async with AsyncSessionLocal() as session:
                stmt = sql_text(
                    "SELECT DISTINCT metadata->>'section_name' AS section, "
                    "metadata->>'course_name' AS item "
                    "FROM documents "
                    "WHERE metadata->>'section_name' IS NOT NULL "
                    "AND metadata->>'section_name' <> '' "
                    "AND metadata->>'course_name' IS NOT NULL "
                    "AND metadata->>'course_name' <> '' "
                    "ORDER BY 1, 2"
                )
                rows = (await session.execute(stmt)).all()
                for r in rows:
                    section_map.setdefault(r.section, [])
                    if r.item not in section_map[r.section]:
                        section_map[r.section].append(r.item)
        except Exception as exc:
            logger.warning(f"Section-map load failed (Postgres): {exc}")
            return {}

        _section_map_cache["map"] = section_map
        _section_map_cache["expires_at"] = now + _COURSE_CACHE_TTL_SECONDS
        return section_map


def _match_section(query: str, sections: list[str]) -> str | None:  # noqa: ARG001
    """REMOVED — replaced by UI-driven section navigation (see /chat/sections
    endpoint + the topic-list button in the chat UI). Kept as a no-op stub only
    if something still imports it; nothing in-tree does."""
    return None


async def _generate_node(state: RAGState, config: RunnableConfig):
    """Single conversational LLM call — the only answer-generating node.

    One CONVERSATIONAL_PROMPT handles everything: greetings, identity, meta-turns
    ("kok gini", "ga nyambung"), chit-chat, and grounded KB answers. Retrieved
    context is injected ONLY when it's actually relevant (the dense-floor gate
    via _route_after_rag passes); for a greeting / off-scope / no-match turn we
    inject NO context, so the model never gets irrelevant chunks forced into a
    casual reply — it just answers conversationally or says it doesn't have that
    info. Memory (STM summary, LTM profile, user prefs) is always injected when
    present. Conciseness + the detail/teach escalation live in the prompt.
    """
    chunks = state.get("retrieved_context") or []
    summary = state.get("conversation_summary") or ""
    profile = state.get("user_profile") or {}
    intent = state.get("intent") or "KNOWLEDGE"

    # Inject context ONLY when retrieval is genuinely relevant. Reuse the
    # dense-floor NOT-FOUND gate: below the floor (greeting/off-scope/no real
    # match) → no context block, and the prompt's <no_context> rules take over.
    has_kb_context = bool(chunks) and _route_after_rag(state) == "generate"

    context_section = ""
    if has_kb_context:
        # Fix A — set/list enumeration ("produk apa aja", "sebutkan semua",
        # "8 prinsip"): the answer oscillated across multi-turn (model defended
        # an earlier wrong count, e.g. "dua doang"). Two mitigations:
        #  (1) surface the most enumerative chunk (the summary/overview that
        #      lists the whole set as bullets) FIRST so the complete list is
        #      salient to a weak model, and
        #  (2) a directive to re-derive the full set from context every turn,
        #      ignoring any count stated in a prior turn.
        import re as _re
        _user_q = (state.get("rewritten_query") or state.get("retrieval_query") or "").lower()
        _is_set_list = bool(_re.search(
            r"\bapa\s*aja\b|\bapa\s*saja\b|\bsebut(?:kan|in|ke)\b|\bsemua\b|"
            r"\bdaftar\b|\blist\b|\b\d+\s+(?:produk|prinsip|value|nilai|jenis|macam|tipe|fitur)\b",
            _user_q,
        ))
        _ordered = chunks
        if _is_set_list and len(chunks) > 1:
            # Bullet-rich chunk = the summary/overview that enumerates the set.
            def _bullet_count(c):
                return len(_re.findall(r"(?m)^\s*[-*]\s", c.get("text", "")))
            _ordered = sorted(chunks, key=_bullet_count, reverse=True)

        # Per-chunk char cap, ATX-heading strip (so an echoed chunk can't render
        # as an <h1>), then a token ceiling on the whole block.
        chunk_char_cap = _settings.lms_chunk_text_max_chars
        context_lines = []
        for i, c in enumerate(_ordered, 1):
            chunk_text = _strip_md_headings_for_context(c.get("text", ""))
            if chunk_char_cap and len(chunk_text) > chunk_char_cap:
                chunk_text = chunk_text[:chunk_char_cap].rstrip() + "…"
            context_lines.append(
                f"[{i}] Course: {c.get('course_name', '?')} (ID:{c.get('course_id', '?')})\n"
                f"{chunk_text}"
            )
        context_str = truncate_to_tokens(
            "\n\n---\n\n".join(context_lines), _settings.max_context_tokens
        )
        context_section = f"\n\n<retrieved_context>\n{context_str}\n</retrieved_context>"
        # NOTE: the set/list enumeration RULE lives in <grounding> in the system
        # prompt (stable, rarely echoed). We deliberately do NOT append a
        # plain-prose directive to the context here — Flash Lite echoed it
        # verbatim to the user ("CATATAN PENTING: ..."). The chunk reordering
        # above (summary-first) is the structural nudge; the prompt carries the
        # instruction.

    # TOPIC_LIST: the user asked what materials/topics exist ("ada materi apa
    # aja"). This is a no-retrieval intent, so inject the GROUND-TRUTH course
    # list from Postgres — otherwise the model invents plausible-sounding topics
    # (the bug). The prompt is told to list ONLY these.
    topics_section = ""
    if intent == "TOPIC_LIST":
        try:
            course_names = await _load_course_names()
        except Exception:
            course_names = []
        if course_names:
            topics_section = (
                "\n\n<available_topics>\n"
                + "\n".join(f"- {c}" for c in course_names)
                + "\n</available_topics>\n"
                "User asked what topics/materials exist. List ONLY the topics in "
                "<available_topics> above, verbatim — do NOT invent or rename any."
            )
        else:
            # Empty list = Postgres load failed or no docs ingested. Without
            # this branch topics_section stays "", _is_grounded falls to False,
            # and the warm chat LLM answers a "what topics?" turn with NO
            # grounding — which fabricates plausible-sounding topics (the
            # "Pinjaman Modal, Cicilan Emas" hallucination). Inject an explicit
            # no-data directive instead; the non-empty string also flips
            # _is_grounded → True so the deterministic (temp 0) client is used.
            topics_section = (
                "\n\n<available_topics>\n(could not load topic list right now)\n"
                "</available_topics>\n"
                "User asked what topics/materials exist, but the list is "
                "unavailable. Say briefly that you can't pull up the topic list "
                "right now and ask them to try again — do NOT invent, guess, or "
                "name ANY topics."
            )

    # Long-term memory (LTM profile)
    ltm_section = ""
    if profile.get("summary"):
        course_names_str = ", ".join(profile.get("course_names", []))
        unanswered = profile.get("unanswered_questions") or []
        history_lines = [
            f"User pernah membahas materi: {course_names_str}",
            f"Konteks sesi sebelumnya: {profile['summary']}",
        ]
        if unanswered:
            history_lines.append(
                "Pertanyaan user yang belum sempat terjawab di sesi lalu: "
                + "; ".join(unanswered)
            )
        ltm_section = "\n\n<user_history>\n" + "\n".join(history_lines) + "\n</user_history>"

    # Short-term rolling summary
    summary_section = ""
    if summary:
        summary_section = f"\n\n<previous_context>\n{summary}\n</previous_context>"

    # Persistent user preferences
    pref_section = ""
    prefs = state.get("user_preferences")
    if prefs:
        pref_lines = []
        if prefs.get("role"):
            pref_lines.append(f"Role/Jabatan User: {prefs['role']}")
        if prefs.get("preferred_tone"):
            pref_lines.append(f"Gaya Bahasa yang Diinginkan: {prefs['preferred_tone']}")
        if prefs.get("formatting_pref"):
            pref_lines.append(f"Format Jawaban: {prefs['formatting_pref']}")
        if prefs.get("custom_instructions"):
            pref_lines.append(f"Instruksi Tambahan: {prefs['custom_instructions']}")
        if pref_lines:
            pref_section = (
                "\n\n<user_preferences>\nSesuaikan jawabanmu dengan profil user berikut:\n"
                + "\n".join(pref_lines)
                + "\n</user_preferences>"
            )

    # Live Moodle profile of the person asking (firstname + custom fields).
    # Lets Ava greet by name and tailor answers to their dept/role/location.
    # Rendered only when at least one field is present (greetings from a
    # tokenless dev session carry nothing).
    user_ctx_section = ""
    uctx = state.get("user_context") or {}
    if uctx:
        ctx_lines = []
        if uctx.get("name"):
            ctx_lines.append(f"Nama: {uctx['name']}")
        if uctx.get("dept"):
            ctx_lines.append(f"Departemen: {uctx['dept']}")
        if uctx.get("position"):
            ctx_lines.append(f"Posisi: {uctx['position']}")
        if uctx.get("grade"):
            ctx_lines.append(f"Grade: {uctx['grade']}")
        if uctx.get("location"):
            ctx_lines.append(f"Lokasi: {uctx['location']}")
        if uctx.get("point"):
            ctx_lines.append(f"Point: {uctx['point']}")
        if ctx_lines:
            user_ctx_section = (
                "\n\n<user_context>\nKamu sedang berbicara dengan user berikut. "
                "Sapa dengan nama depannya bila relevan dan sesuaikan jawaban "
                "dengan konteksnya:\n"
                + "\n".join(ctx_lines)
                + "\n</user_context>"
            )

    dynamic_tail = f"{user_ctx_section}{pref_section}{ltm_section}{summary_section}{topics_section}{context_section}".strip()

    # Temperature split (no extra tokens — just which pre-built client we call):
    #   - GROUNDED turn (KB <context> present, or a TOPIC_LIST with the real
    #     course list) → temp 0.0 (get_generate_llm) so factual / enumeration
    #     answers are deterministic and consistent turn-to-turn. Fixes the
    #     "produk apa aja" giving a different list each time at temp 0.4.
    #   - CONVERSATIONAL turn (greeting, identity, vent, meta, no context) →
    #     temp 0.4 (get_chat_llm) so chit-chat stays warm and natural.
    #   - COACHING turn → temp 0.4 (get_chat_llm): the Socratic guiding question
    #     needs warmth/variation to feel like a trainer, not a form. Faithfulness
    #     is still enforced by SOCRATIC_PROMPT's grounding rules + the dense-floor
    #     gate (context injected only when relevant).
    # All clients share the same model/provider/streaming/usage flags. Coaching
    # uses a DIFFERENT cached system prefix (SOCRATIC_PROMPT) — that's a separate
    # prompt-cache entry, still cacheable, just not shared with the conv prefix.
    is_coaching = intent == "COACHING"
    _is_grounded = (has_kb_context or bool(topics_section)) and not is_coaching
    llm = get_generate_llm() if _is_grounded else get_chat_llm()
    windowed_messages = _window_generate_history(
        list(state["messages"]),
        max_fresh_turns=_settings.max_fresh_turns,
        max_ai_chars=_settings.max_history_ai_chars,
    )
    # Static prompt wrapped in a cache_control breakpoint so OpenRouter/Vertex
    # serves it from the provider prefix-cache on the 2nd+ call. Dynamic per-turn
    # context lives in a separate HumanMessage so the cached prefix is byte-stable.
    #
    # Per-intent prompt selection (Option C — saves ~900 tok on chit-chat turns):
    #   - COACHING → SOCRATIC_PROMPT (full Socratic scaffolding)
    #   - KNOWLEDGE / TOPIC_LIST → CONVERSATIONAL_PROMPT (full grounding rules)
    #   - GREETING / AMBIGUOUS / OFF_SCOPE → CHIT_CHAT_PROMPT (minimal, no KB)
    if is_coaching:
        system_prompt_text = SOCRATIC_PROMPT
    elif intent in ("GREETING", "AMBIGUOUS", "OFF_SCOPE"):
        system_prompt_text = CHIT_CHAT_PROMPT
    else:
        # KNOWLEDGE, TOPIC_LIST
        system_prompt_text = CONVERSATIONAL_PROMPT
    system_msg = SystemMessage(content=[
        {"type": "text", "text": system_prompt_text,
         "cache_control": {"type": "ephemeral", "ttl": "1h"}},
    ])
    # Only inject the dynamic context message when there's actually something in
    # it (a greeting with no context/memory shouldn't get an empty block).
    msgs: list = [system_msg]
    if dynamic_tail:
        msgs.append(HumanMessage(content=dynamic_tail))
    msgs += windowed_messages

    response = await llm.ainvoke(msgs, config=config)
    _log_cache_usage(response, "generate")

    # Defensive anti-leak: strip any <retrieved_context>/<user_history>/etc. block
    # the model may have echoed verbatim (also prevents `# Heading` chunks from
    # rendering as 4x font). Prompt-level OUTPUT_CONTRACT handles the 99% case.
    raw = response.content if hasattr(response, "content") else str(response)
    if isinstance(raw, str):
        cleaned = _sanitize_answer(raw)
        if cleaned != raw:
            logger.warning(
                "generate_node: stripped leaked instruction block from LLM output "
                f"(orig_len={len(raw)} clean_len={len(cleaned)})"
            )
            response.content = cleaned

    return {"messages": [response]}


# ─── Routing ─────────────────────────────────────────────────────────────────

def _route_by_intent(state: RAGState) -> str:
    return state.get("intent") or "KNOWLEDGE"


def _route_after_rag(state: RAGState) -> str:
    """Decide whether to call the LLM or short-circuit when retrieval is weak.

    Applies to both KNOWLEDGE and COACHING — a Socratic guiding question must
    be grounded in the KB just like a factual answer, so COACHING gets NO
    special bypass: if retrieval is below the floor, context is withheld and the
    prompt's no-context / grounding rules take over ("Aku belum nemu ini di
    materiku") rather than inventing a question around nothing.

    The gate is a MANDATORY DENSE FLOOR (not an OR). bge-m3 dense cosine cleanly
    separates scope on this KB — off-scope tops out ~0.36, in-scope floors ~0.50
    — so dense alone is the discriminator:
      - `dense_score` — raw dense cosine [0, 1], an ABSOLUTE semantic signal.
        Calibrated: off-scope ceiling ≈ 0.36, in-scope floor ≈ 0.50; floor set
        at 0.40 (safe band [0.40, 0.45], leaning in-scope-protective).
    Sparse (raw BM25) is NOT consulted in normal operation: on a small KB it does
    NOT separate scope — off-scope function words ("gimana", "di", "siapa")
    accumulate BM25 (off-scope sparse 8.56 outscores in-scope "Poket" 6.23), so an
    OR-rescue on sparse re-admits off-scope. The fused `score`/`hybrid_score` is
    min-max normalized per-query (top hit ≈ 1.0 on every query), so it can't gate
    a global miss — the raw dense signal can. Sparse is read ONLY in the C5
    degraded window below (embedding outage), as a fail-open backstop.

    C4 — the gate reads POOL-LEVEL maxes (`pool_max_dense`/`pool_max_sparse`,
    set by rag_node over the full fetch_k pool BEFORE the top-k slice), not the
    per-chunk maxes of the returned top-k. A chunk with the highest raw dense
    cosine can rank below the final top-k by FUSED score (fusion blends in
    normalized sparse) and get sliced off — gating on the post-slice chunks
    would then read an artificially low max and emit a FALSE NOT-FOUND. Falls
    back to the per-chunk computation if pool stats are absent (defensive).

    C5 — when retrieval degraded to sparse-only (embedding outage,
    `dense_retrieval_ok` is False), the dense signal is MISSING, not low. We
    must not let an absent dense score force a NOT-FOUND; the gate runs on
    sparse alone in that window.
    """
    chunks = state.get("retrieved_context") or []
    if not chunks:
        return "low_relevance"

    # C4: prefer pool-level maxes computed over the full fetch_k pool. Fall back
    # to per-chunk maxes (legacy behaviour) only when pool stats are unavailable.
    pool_max_dense = state.get("pool_max_dense")
    pool_max_sparse = state.get("pool_max_sparse")
    if pool_max_dense is None and pool_max_sparse is None:
        dense_scores = [v for c in chunks if isinstance((v := c.get("dense_score")), (int, float))]
        sparse_scores = [v for c in chunks if isinstance((v := c.get("sparse_score")), (int, float))]
        if not dense_scores and not sparse_scores:
            return "low_relevance"
        max_dense = max(dense_scores) if dense_scores else 0.0
        max_sparse = max(sparse_scores) if sparse_scores else 0.0
    else:
        max_dense = float(pool_max_dense or 0.0)
        max_sparse = float(pool_max_sparse or 0.0)

    # C5: dense degraded to sparse-only — judge on sparse alone, don't let the
    # missing dense signal (0.0) manufacture a NOT-FOUND.
    dense_retrieval_ok = state.get("dense_retrieval_ok")
    if dense_retrieval_ok is False:
        if max_sparse >= _settings.kb_min_sparse_score:
            return "generate"
        logger.info(
            f"Sparse-only retrieval below sparse gate — skipping generate_node "
            f"(sparse={max_sparse:.4f} < {_settings.kb_min_sparse_score}, dense unavailable)"
        )
        return "low_relevance"

    # Normal operation: DENSE is the mandatory floor. Sparse is NOT consulted —
    # raw BM25 does not separate scope on a small KB (off-scope function-word
    # matches outscore real entities), so an OR-rescue on sparse re-admits
    # off-scope. The clean dense gap (off-scope ≤~0.36, in-scope ≥~0.50) is the
    # gate.
    if max_dense < _settings.kb_min_dense_score:
        # NEAR-MISS monitor: dense just below the floor. Could be a terse
        # entity/acronym wrongly rejected — instrument so ops can pull these and
        # decide whether a corroboration tier is needed. Does NOT change routing.
        if max_dense >= _settings.kb_min_dense_score - 0.05:
            logger.info(
                f"NEAR-MISS reject (dense just below floor) — "
                f"dense={max_dense:.4f} < {_settings.kb_min_dense_score}, "
                f"sparse={max_sparse:.4f}"
            )
            _emit_sparse_only_passthrough()
        logger.info(
            f"Dense below floor — NOT-FOUND "
            f"(pool dense={max_dense:.4f} < {_settings.kb_min_dense_score}, "
            f"pool sparse={max_sparse:.4f} ignored)"
        )
        return "low_relevance"
    return "generate"


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def _build_agent_graph():
    """Build and compile the minimal conversational RAG StateGraph.

    Collapsed from the old 9-node / 7-intent router to 4 nodes. Routing by the
    regex Tier-1 label set in _pre_processor (no LLM):
        START → pre_processor → MALICIOUS                    → malicious      → END
                              → GREETING/AMBIGUOUS/OFF_SCOPE/TOPIC_LIST
                                                              → generate_node → END  (no retrieval)
                              → KNOWLEDGE → rag_node          → generate_node → END

    Chit-chat / no-lookup intents skip retrieval entirely and go straight to the
    conversational generate node with NO <context> — so a greeting or a vague
    "info dong" never gets an irrelevant chunk dumped on it, and the prompt asks
    a clarifying question instead of guessing. Only a real KNOWLEDGE question
    retrieves. generate_node is the single conversational LLM call; the canned
    handlers (greeting/ambiguity/off_scope/topic_list/low_relevance) are gone —
    their behavior lives in CONVERSATIONAL_PROMPT.
    """
    builder = StateGraph(RAGState)

    # Nodes
    builder.add_node("pre_processor", _pre_processor)
    builder.add_node("malicious", _handle_malicious)
    builder.add_node("rag_node", _rag_node)
    builder.add_node("generate_node", _generate_node)

    # Edges
    builder.add_edge(START, "pre_processor")
    builder.add_conditional_edges(
        "pre_processor",
        _route_by_intent,
        {
            "MALICIOUS": "malicious",
            # No-lookup intents → straight to the conversational LLM, no retrieval.
            "GREETING": "generate_node",
            "AMBIGUOUS": "generate_node",
            "OFF_SCOPE": "generate_node",
            "TOPIC_LIST": "generate_node",
            # Real question → retrieve first.
            "KNOWLEDGE": "rag_node",
            # Coaching (Socratic) also retrieves first — the guiding question
            # must be grounded in the KB, so it flows through rag_node too.
            "COACHING": "rag_node",
        },
    )
    builder.add_edge("malicious", END)
    # rag_node ALWAYS flows to generate_node — the dense-floor NOT-FOUND gate is
    # applied inside generate_node (context injected only when relevant), so
    # there's no separate low_relevance dead-end. _route_after_rag is still used
    # by chat.py for cache-write gating and by generate_node for context gating.
    builder.add_edge("rag_node", "generate_node")
    builder.add_edge("generate_node", END)

    return builder.compile()


@lru_cache(maxsize=1)
def get_rag_graph():
    """Return the singleton compiled RAG graph."""
    return _build_agent_graph()
