// ============================================================
// AUTO INJECT HTML ke body (untuk Moodle integration)
// ============================================================
(function injectChatWidget() {
    // Cek apakah sudah ada (hindari duplikat)
    if (document.getElementById("chat-toggle")) return;

    // Inject Inter font (widget hidup di halaman Moodle yang tidak punya font ini)
    if (!document.getElementById("ava-inter-font")) {
        const fontLink = document.createElement("link");
        fontLink.id = "ava-inter-font";
        fontLink.rel = "stylesheet";
        fontLink.href = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap";
        document.head.appendChild(fontLink);
    }

    const html = `
        <button id="chat-toggle">
            <i class="fas fa-comment-dots chat-toggle-icon icon-messenger"></i>
            <i class="fas fa-xmark chat-toggle-icon icon-close"></i>
        </button>

        <!-- Pending-response popup (see index.html for full comment) -->
        <div id="ava-popup" class="ava-popup" hidden>
            <div class="ava-popup-body">
                <div class="ava-popup-text"></div>
                <button class="ava-popup-close" title="Dismiss">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        </div>

        <div id="chat-box" class="animate__animated">
            <div id="chat-header">
                <div class="header-info">
                    <div class="online-dot"></div>
                    <div style="display:flex; flex-direction:column;">
                        <span>AI Trainer</span>
                        <small style="font-size:11px; opacity:.8;">AI Trainer can make mistake</small>
                    </div>
                </div>
                <div style="display:flex; gap:20px; align-items:center;">
                    <button id="coach-cta-btn" class="coach-cta-btn" title="Aktifkan Coaching Mode">
                        <i class="fas fa-lightbulb"></i> Coaching
                    </button>
                    <i class="fas fa-trash-alt header-icon" onclick="clearChat()" title="Clear chat" style="cursor:pointer; font-size:14px; opacity:0.8;"></i>
                </div>
            </div>

            <div id="chat-messages"></div>
            <div id="chat-input">
                <button id="topics-btn" class="topics-btn" onclick="openSectionPanel()" title="Daftar topik">
                    <i class="fas fa-book"></i>
                </button>
                <textarea id="prompt" rows="1" placeholder="Ketik pesan..." onkeydown="handleKey(event)"></textarea>
                <button id="send-btn" class="send-btn" onclick="handleSendClick()">
                    <i class="fas fa-paper-plane"></i>
                </button>
            </div>
        </div>

        <!-- Onboarding tutorial elements (see index.html for full comment) -->
        <div id="ava-tour-overlay" class="ava-tour-overlay" hidden></div>
        <div id="ava-tour-cutout" class="ava-tour-cutout" hidden></div>
        <div id="ava-tour-tooltip" class="ava-tour-tooltip" hidden>
            <strong id="ava-tour-title"></strong>
            <div id="ava-tour-body"></div>
            <div class="ava-tour-actions">
                <div class="ava-tour-dots" id="ava-tour-dots"></div>
                <div>
                    <button id="ava-tour-skip" class="ava-tour-btn ava-tour-btn-ghost">Skip</button>
                    <button id="ava-tour-next" class="ava-tour-btn ava-tour-btn-primary">Lanjut →</button>
                </div>
            </div>
        </div>
    `;

    const wrapper = document.createElement("div");
    wrapper.innerHTML = html;
    document.body.appendChild(wrapper);
})();

// ============================================================
// INIT — tunggu DOM siap
// ============================================================
const chatBox = document.getElementById("chat-box");
const messages = document.getElementById("chat-messages");
const textarea = document.getElementById("prompt");
let introduced = false;
let isStreaming = false; // Prevent double-sends during streaming
let currentAbortController = null;

// ── Kebab menu config: extensible for future features (roleplay, etc.) ──
// Each item:
//   id      unique key
// Add event listener for the CTA button
const ctaBtn = document.getElementById("coach-cta-btn");
if (ctaBtn) {
    ctaBtn.addEventListener("click", () => {
        const isCurrentlyOn = document.body.classList.contains("coaching-active");
        setCoaching(!isCurrentlyOn, true);
    });
}

// ── Pending popup click handlers ──
// Click the popup body → open the chatbox (the answer is already streamed
// into the chat, but the user might have closed it mid-stream — re-opening
// shows the full bubble). Click × → just dismiss.
document.addEventListener("click", (e) => {
    const popup = document.getElementById("ava-popup");
    if (!popup || popup.hidden) return;

    // × button: dismiss only, don't open chat
    if (e.target.closest(".ava-popup-close")) {
        e.stopPropagation();
        dismissPendingPopup();
        return;
    }
    // Anywhere else on popup: open chat + dismiss
    if (popup.contains(e.target)) {
        if (_isChatboxHidden()) {
            toggleChat();
        } else {
            dismissPendingPopup();
        }
    }
});

// Anchor the coaching tab to the chatbox's left edge. Re-run on resize
// so the tab follows the chatbox on desktop ↔ mobile width changes.


function setSendButtonState(streaming) {
    const btns = document.querySelectorAll(".send-btn");
    btns.forEach(btn => {
        const icon = btn.querySelector("i");
        if (icon) {
            if (streaming) {
                icon.className = "fas fa-stop"; // Stop icon
                btn.classList.add("cancel-mode");
            } else {
                icon.className = "fas fa-paper-plane"; // Send icon
                btn.classList.remove("cancel-mode");
            }
        }
    });
}

function handleSendClick() {
    if (isStreaming) {
        // Cancel the ongoing request
        if (currentAbortController) {
            currentAbortController.abort();
            currentAbortController = null;
        }
        isStreaming = false;
        setSendButtonState(false);
    } else {
        send();
    }
}

/* AKTIFKAN PULSE */
const toggleBtn = document.getElementById("chat-toggle");
if (toggleBtn) {
    toggleBtn.classList.add("pulse");
    toggleBtn.onclick = toggleChat;
}

// ============================================================
// TOGGLE CHAT
// ============================================================
// ============================================================
// COACHING MODE — single source of truth for the slider + chips
// ============================================================
// setCoaching(on, showMsg): sets the flag read by send() (coaching_mode in
// the request body), syncs the CTA button, and — when showMsg — injects an
// awareness message so the user SEES the mode change. Both on AND off respond,
// so the switch never flips silently.
function setCoaching(on, showMsg) {
    window.COACHING_MODE = !!on;
    document.body.classList.toggle("coaching-active", !!on);
    
    // Sync the CTA button
    const ctaBtn = document.getElementById("coach-cta-btn");
    if (ctaBtn) {
        if (on) {
            ctaBtn.classList.add("active");
            ctaBtn.innerHTML = '<i class="fas fa-times"></i> Exit Coaching';
        } else {
            ctaBtn.classList.remove("active");
            ctaBtn.innerHTML = '<i class="fas fa-lightbulb"></i> Coaching';
        }
    }

    if (showMsg) {
        if (on) {
            addMessage(
                "Oke, kita brainstorming bareng ya. Aku pandu kamu sampai nemu jawabannya sendiri. Mau mulai dari mana, materi Amarthapedia atau soal kerjaan kamu di Amartha?",
                "ai"
            );
        } else {
            addMessage("Yuk, tanya apa aja soal materi Amarthapedia, aku temani kamu belajar.", "ai");
        }
    }
}

// Welcome-screen chip: "Topik" → fetch the topic list straight from the
// instant endpoint (Postgres, no LLM) and render it client-side, so the list
// appears immediately instead of waiting on the chat pipeline's generate step.
async function chipTopik() {
    removeWelcome();
    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = { "ngrok-skip-browser-warning": "true" };
    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }
    try {
        const res = await fetch(`${baseUrl}/api/v1/chat/topics`, { method: "GET", headers });
        const data = await res.json();
        const topics = (data && data.topics) || [];
        if (topics.length) {
            const list = topics.map(t => `* ${t}`).join("\n");
            addMessage("Topik yang tersedia di Amarthapedia:\n\n" + list, "ai");
        } else {
            addMessage("Belum ada topik yang bisa aku tampilkan saat ini.", "ai");
        }
    } catch (e) {
        console.error("chipTopik failed:", e);
        addMessage("Aku belum bisa nampilin daftar topik sekarang. Coba lagi ya.", "ai");
    }
}

// Welcome-screen chip: "Coaching" → flip the slider ON and show the canned
// guiding prompt instantly (client-side, no backend round-trip).
function chipCoaching() {
    setCoaching(true, true);
}

// ── Topic-list button → in-chatbox section/item picker ───────────────────────
// Fetches /chat/sections (deterministic, no LLM) and renders an accordion INSIDE
// #chat-box (clipped by overflow:hidden, like the delete modal). Click a section
// to expand its items; click an item to send "jelaskan tentang <item>" as a
// normal KNOWLEDGE query. Replaces the old fragile free-text section parsing.
function _esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

async function openSectionPanel() {
    if (document.getElementById("ava-section-panel")) return;  // already open
    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = { "ngrok-skip-browser-warning": "true" };
    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }
    let sections = {};
    try {
        const res = await fetch(`${baseUrl}/api/v1/chat/sections`, { method: "GET", headers });
        const data = await res.json();
        sections = (data && data.sections) || {};
    } catch (e) {
        console.error("openSectionPanel fetch failed:", e);
    }

    const overlay = document.createElement("div");
    overlay.id = "ava-section-panel";
    overlay.className = "ava-panel-overlay";

    const names = Object.keys(sections);
    let rows = "";
    if (names.length) {
        rows = names.map(sec => {
            const items = (sections[sec] || []).map(it =>
                `<button class="ava-panel-item" data-item="${_esc(it)}">${_esc(it)}</button>`
            ).join("");
            return `<div class="ava-panel-section">
                <button class="ava-panel-sec-head"><span>${_esc(sec)}</span><i class="fas fa-chevron-down"></i></button>
                <div class="ava-panel-items">${items}</div>
            </div>`;
        }).join("");
    } else {
        rows = `<div class="ava-panel-empty">Belum ada topik yang bisa ditampilkan.</div>`;
    }

    overlay.innerHTML = `
        <div class="ava-panel-card">
            <div class="ava-panel-head">
                <span>Daftar Topik</span>
                <i class="fas fa-times ava-panel-close" title="Tutup"></i>
            </div>
            <div class="ava-panel-body">${rows}</div>
        </div>`;

    const close = () => overlay.remove();
    overlay.querySelector(".ava-panel-close").onclick = close;
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    // Accordion: toggle a section open/closed.
    overlay.querySelectorAll(".ava-panel-sec-head").forEach(h => {
        h.onclick = () => h.parentElement.classList.toggle("open");
    });
    // Click an item → close panel + ask about it as a normal question.
    overlay.querySelectorAll(".ava-panel-item").forEach(b => {
        b.onclick = () => {
            const item = b.getAttribute("data-item");
            close();
            send(`jelaskan tentang ${item}`);
        };
    });
    chatBox.appendChild(overlay);
}

// ── Auto-hook: OFFER coaching after a reflective question ────────────────────
// Detects a diagnostic/"how-should-I" question about the user's OWN work and
// (when mentoring is OFF) appends a soft OFFER below the answer. It never
// auto-activates — the user must click "Ya, pandu aku" to opt in (which turns
// the switch green). Frontend-only: the answer already streamed normally.
function _looksReflective(t) {
    if (!t) return false;
    const s = " " + t.toLowerCase() + " ";
    const why = /\b(kok|kenapa|mengapa|knp|ngapa|napa)\b/.test(s);
    const work = /\b(aku|ku|saya|mitra|nasabah|target|tim|point|portfolio|portofolio|repayment|tagih|ditagih|kabur|nunggak|setoran|angsuran)\b/.test(s);
    if (why && work) return true;
    // "gimana caranya aku ...", "gimana aku bisa/harus ..."
    if (/\b(gimana|gmn|gmana|bagaimana)\b[^.?!\n]{0,25}\b(aku|ku|saya)\b/.test(s)) return true;
    return false;
}

function removeCoachOffer() {
    const el = document.getElementById("ava-coach-offer");
    if (el) el.remove();
}

// backendSuggest: the server's suggest_coaching flag (true/false), or null/
// undefined when absent. When present it's AUTHORITATIVE (semantic affinity OR
// topic-streak); the regex _looksReflective is only the fallback when the
// backend didn't send a signal (e.g. fallback/non-stream path or older backend).
// topic: when the server's offer was triggered by a topic streak, the dominant
// topic name — used to word the offer around that topic.
function maybeOfferCoaching(userText, backendSuggest, topic) {
    if (window.COACHING_MODE) return;                        // already on
    if (document.getElementById("ava-coach-offer")) return;  // one at a time
    const show = (backendSuggest === true || backendSuggest === false)
        ? backendSuggest
        : _looksReflective(userText);
    if (!show) return;
    window._lastReflectiveQ = userText;  // remembered for the accept handler
    const wrap = document.createElement("div");
    wrap.id = "ava-coach-offer";
    wrap.className = "ava-coach-offer animate__animated animate__fadeIn animate__faster";
    const text = topic
        ? 'Kayaknya kamu lagi tertarik banget sama <b>' + _esc(topic) +
        '</b> nih. Mau brainstorming bareng?'
        : 'Mau kita bahas ini lebih dalam bareng-bareng? Aku temani kamu brainstorming.';
    wrap.innerHTML =
        '<span class="ava-offer-text">' + text + '</span>' +
        '<button class="ava-offer-btn" onclick="acceptCoachingOffer()"><i class="fas fa-graduation-cap"></i> Gas Coaching</button>';
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
}

// ── Trigger #3 (explicit): user literally asks to be coached/mentored ────────
// "coaching", "mentor", "aku mau dimentorin", "pandu aku", "mode coaching",
// "belajar bareng" → offer the mode immediately instead of answering in normal
// mode. Bare "coach"/"mentor" counts, EXCEPT when the turn is a content question
// ABOUT mentors/coaches as a topic ("apa itu mentor", "tugas mentor apa") —
// those must be answered, not intercepted. Generic "ajarin aku X" is NOT matched
// (normal teach request — Ava already teaches by default).
function _wantsCoaching(t) {
    if (!t) return false;
    const s = " " + t.toLowerCase() + " ";
    // Specific mode-request phrasings — always intercept.
    if (/\b(mode (coaching|coach|mentor)|(coaching|coach|mentor) mode)\b/.test(s)) return true;
    if (/\b(aktif(in|kan)?|nyalain|hidupin)\s+(mode\s+)?(coaching|coach|mentor)\b/.test(s)) return true;
    if (/\b(pandu|bimbing|tuntun)\s+(aku|saya|gw|gue|ku)\b/.test(s)) return true;
    if (/\b(belajar|ngulik)\s+bareng\b/.test(s)) return true;
    // Bare "coach"/"mentor"/"mentorin" → mode request, UNLESS it's a definitional/
    // content question about mentors/coaches as a topic.
    if (/\b(coaching|coach|(di)?mentor(in|i|kan)?)\b/.test(s)) {
        const aboutTopic =
            /\b(apa\s*itu|apa\s*sih|apa|siapa|tugas|peran|fungsi|gimana|bagaimana|jelas(in|kan)?)\b[^?\n]{0,20}\b(coach|mentor)\w*\b/.test(s)
            || /\b(coach|mentor)\w*\b[^?\n]{0,12}\b(itu|tuh)\s+(apa|siapa|gimana)\b/.test(s);
        return !aboutTopic;
    }
    return false;
}

// Pull the TOPIC out of an explicit request so accepting re-asks it Socratically
// ("aku mau dimentorin soal cara nagih" → "cara nagih"). "" when no topic given.
function _stripCoachingPhrase(t) {
    let s = (t || "");
    s = s.replace(/\b(coaching|coach|(di)?mentor(in|i|kan)?)\b/gi, " ");
    s = s.replace(/\b(mode (coaching|coach|mentor)|(coaching|coach|mentor) mode)\b/gi, " ");
    s = s.replace(/\b(aktif(in|kan)?|nyalain|hidupin)\b/gi, " ");
    s = s.replace(/\b(pandu|bimbing|tuntun)\b/gi, " ");
    s = s.replace(/\b(belajar|ngulik)\s+bareng\b/gi, " ");
    s = s.replace(/\b(aku|saya|gw|gue|ku)\b/gi, " ");
    s = s.replace(/\b(mau|pengen|pingin|pgn|minta|tolong|dong|donk|ya|nih|deh|coba)\b/gi, " ");
    s = s.replace(/\b(soal|tentang|mengenai|terkait|buat|untuk|ttg)\b/gi, " ");
    s = s.replace(/\bstep by step\b/gi, " ");
    s = s.replace(/[?!.]+/g, " ").replace(/\s+/g, " ").trim();
    return s;
}

// Render the offer card with custom text + the standard accept button.
function _renderCoachOffer(innerHtml) {
    if (document.getElementById("ava-coach-offer")) return;
    const wrap = document.createElement("div");
    wrap.id = "ava-coach-offer";
    wrap.className = "ava-coach-offer animate__animated animate__fadeIn animate__faster";
    wrap.innerHTML =
        '<span class="ava-offer-text">' + innerHtml + '</span>' +
        '<button class="ava-offer-btn" onclick="acceptCoachingOffer()"><i class="fas fa-graduation-cap"></i> Gas Coaching</button>';
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
}

// Offer shown when the user EXPLICITLY asked to be coached. Worded as an
// acknowledgement so it doesn't feel like the bot ignored the request.
function offerExplicitCoaching() {
    _renderCoachOffer("Siap! Aku bisa pandu kamu mikir step by step. Aktifin mode Coaching sekarang?");
}

function _isChatboxHidden() {
    return chatBox.style.display === "none" || chatBox.style.display === "";
}

// message) and re-ask their last question in mentoring mode so the answer
// continues straight into Socratic coaching ON THAT topic — not a reset to
// "apa yang bikin kamu bingung?". skipBubble: the question is already shown
// above, don't duplicate it.
function acceptCoachingOffer() {
    removeCoachOffer();
    setCoaching(true, false);   // green slider, no canned message
    const q = window._lastReflectiveQ;
    if (q) {
        window._lastReflectiveQ = null;
        send(q, { skipBubble: true });
    } else {
        // No remembered question (shouldn't happen from the offer) — fall back
        // to the canned guiding prompt.
        addMessage("Oke, aku pandu kamu belajar ya. Apa yang lagi pengen kamu ulik?", "ai");
    }
}

// ── Mirror of the coaching offer: OFFER to leave Coaching → back to Mentoring ──
// Fires when the backend signals the Socratic loop is done (coaching_done): Ava
// delivered a wrap-up/direct answer instead of a guiding question. Clicking
// flips the toggle OFF, exactly symmetric to the "Gas Coaching" offer flipping
// it ON. The signal is computed server-side (gated on the real COACHING intent),
// so no marker ever streams to the user — nothing to leak.
function maybeOfferBackToMentoring(coachingDone) {
    if (!window.COACHING_MODE) return;                       // only while coaching ON
    if (document.getElementById("ava-coach-offer")) return;  // one offer card at a time
    if (!coachingDone) return;                               // still in the loop
    const wrap = document.createElement("div");
    wrap.id = "ava-coach-offer";
    wrap.className = "ava-coach-offer animate__animated animate__fadeIn animate__faster";
    wrap.innerHTML =
        '<span class="ava-offer-text">Kayaknya kita udah kelar ngulik bareng nih. Mau aku balik jawab langsung aja?</span>' +
        '<button class="ava-offer-btn" onclick="acceptBackToMentoring()"><i class="fas fa-comments"></i> Balik ke Mentoring</button>';
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;
}

function acceptBackToMentoring() {
    removeCoachOffer();
    setCoaching(false, true);   // flip slider off + the canned "balik jawab langsung" message
}

// ── Pending-response popup ────────────────────────────────────────────────────
// Shown ABOVE the FAB when the AI finishes answering while the chatbox is
// closed. Single slot: each new pending response REPLACES the previous preview
// (no stacking, no queue). FAB pulse intensifies (6s → 1.8s) so the
// "something's waiting" cue is multi-channel.
//
// Triggers:
//   - SSE `done` event fires while chatbox is hidden → showPendingPopup(text)
//   - User clicks popup (anywhere) OR × button → open chatbox + dismiss
//   - 12s idle → auto-dismiss (popup fades, FAB returns to normal pulse)
//   - User manually opens chatbox (toggleChat) → also dismiss in case still up
const POPUP_AUTO_HIDE_MS = 12000;
let _popupHideTimer = null;

function _stripMarkdownForPreview(text) {
    // Crude but safe enough for a 2-line preview. Strips code fences, links,
    // headers, list bullets, bold/italic markers. We DON'T use marked.parse()
    // here because we'd be parsing into HTML and the popup-text is plain text
    // (we want it to inherit font + word-break from the popup, not bubble styles).
    if (!text) return "";
    return text
        .replace(/```[\s\S]*?```/g, "")           // code blocks
        .replace(/`([^`]+)`/g, "$1")              // inline code
        .replace(/!\[[^\]]*\]\([^)]*\)/g, "")     // images
        .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")  // links → text only
        .replace(/^#+\s+/gm, "")                  // headers
        .replace(/^\s*[-*+]\s+/gm, "")            // list bullets
        .replace(/^\s*\d+\.\s+/gm, "")            // ordered list
        .replace(/\*\*([^*]+)\*\*/g, "$1")        // bold
        .replace(/\*([^*]+)\*/g, "$1")            // italic
        .replace(/__([^_]+)__/g, "$1")            // bold (underscore)
        .replace(/_([^_]+)_/g, "$1")              // italic (underscore)
        .replace(/\s+/g, " ")
        .trim();
}

function showPendingPopup(fullText) {
    const popup = document.getElementById("ava-popup");
    const fab = document.getElementById("chat-toggle");
    if (!popup || !fab) return;

    // Render preview (strip markdown, clamp via CSS line-clamp to 2 lines)
    const preview = _stripMarkdownForPreview(fullText);
    const textEl = popup.querySelector(".ava-popup-text");
    if (textEl) textEl.textContent = preview || "(Ava udah jawab — buka chat buat liat)";

    // Reveal with IN animation. If was hidden, set hidden=false then re-add class.
    popup.classList.remove("fadeout");
    popup.hidden = false;
    // Force reflow so animation restarts cleanly on re-show
    void popup.offsetWidth;

    // Intensify FAB pulse (6s → 1.8s). This also overrides the regular
    // .pulse class — they're mutually exclusive, .intensify wins.
    fab.classList.remove("pulse");
    fab.classList.add("intensify");

    // Reset the auto-hide timer
    if (_popupHideTimer) clearTimeout(_popupHideTimer);
    _popupHideTimer = setTimeout(() => {
        dismissPendingPopup();
    }, POPUP_AUTO_HIDE_MS);
}

function dismissPendingPopup() {
    const popup = document.getElementById("ava-popup");
    const fab = document.getElementById("chat-toggle");
    if (!popup || popup.hidden) return;

    if (_popupHideTimer) {
        clearTimeout(_popupHideTimer);
        _popupHideTimer = null;
    }

    // Fade out then hide
    popup.classList.add("fadeout");
    setTimeout(() => {
        popup.classList.remove("fadeout");
        popup.hidden = true;
    }, 220);

    // Restore normal FAB pulse
    if (fab) {
        fab.classList.remove("intensify");
        // Re-add the standard pulse so the icon feels "alive" again
        fab.classList.add("pulse");
    }
}

function toggleChat() {
    const toggleBtn = document.getElementById("chat-toggle");

    if (chatBox.style.display === "none" || chatBox.style.display === "") {
        // ===== OPEN =====
        // Opening the chatbox also dismisses any pending popup — the user is
        // looking at the full chat now, the popup is redundant.
        dismissPendingPopup();

        chatBox.style.display = "flex";
        chatBox.classList.remove("animate__fadeOutDown");
        chatBox.classList.add("animate__fadeInUp");

        // FAB stays visible but morphs into a CLOSE (X) icon. The morph
        // is driven by .chat-open class on <body> (see CSS — both icons
        // cross-fade + rotate). Hide the unread badge.
        document.body.classList.add("chat-open");
        const badge = document.getElementById("chat-badge");
        if (badge) badge.style.display = "none";

        if (!introduced) {
            loadHistory();
            introduced = true;
        }
    } else {
        // ===== CLOSE =====
        chatBox.classList.remove("animate__fadeInUp");
        chatBox.classList.add("animate__fadeOutDown");

        setTimeout(() => {
            chatBox.style.display = "none";
            // FAB morphs back to messenger icon + resume the pulse loop
            // so the user notices the (now-closed) chat is still available.
            document.body.classList.remove("chat-open");
            toggleBtn.classList.add("pulse");
        }, 500);
    }
}

// ============================================================
// INTRO — welcome screen terpusat (sapa user pakai nama Moodle)
// ============================================================

/**
 * Greeting berbasis waktu lokal browser (dihitung fresh tiap dibuka).
 *   5–10  → Selamat pagi
 *   11–14 → Selamat siang
 *   15–17 → Selamat sore
 *   else  → Selamat malam
 */
function getGreeting() {
    const h = new Date().getHours();
    if (h >= 5 && h <= 10) return "Selamat pagi";
    if (h >= 11 && h <= 14) return "Selamat siang";
    if (h >= 15 && h <= 17) return "Selamat sore";
    return "Selamat malam";
}

/** Hapus welcome screen kalau ada (dipanggil saat pesan pertama muncul). */
function removeWelcome() {
    const el = document.getElementById("ava-welcome");
    if (el) el.remove();
}

function showIntro() {
    // Hindari duplikat welcome
    if (document.getElementById("ava-welcome")) return;
    // Welcome = empty-state ONLY. Kalau sudah ada bubble chat (mis. user klik
    // chip Topik/Mentoring lalu close+open), JANGAN munculkan welcome lagi di
    // bawah konten — itu bug "Selamat siang muncul lagi".
    if (messages.querySelector(".msg")) return;

    const nama = (typeof MOODLE_USER_NAME !== 'undefined' && MOODLE_USER_NAME)
        ? MOODLE_USER_NAME.split(' ')[0]
        : 'A-Team';

    const welcome = document.createElement("div");
    welcome.id = "ava-welcome";
    welcome.className = "animate__animated animate__fadeIn animate__faster";
    welcome.innerHTML = `
        <div style="text-align: center; padding: 20px 0;">
            <h2 class="ava-welcome-title">${getGreeting()}, ${nama}</h2>
            <p class="ava-welcome-subtitle">Ada yang bisa aku bantu hari ini terkait materi Amarthapedia?</p>
        </div>
    `;

    messages.appendChild(welcome);
}

async function loadHistory() {

    setTimeout(showIntro, 100);

    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    };

    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }

    try {
        const sessionId = getSessionId();
        const res = await fetch(`${baseUrl}/api/v1/chat/history/${sessionId}`, {
            method: "GET",
            headers: headers
        });

        if (!res.ok) throw new Error("No history found");

        const history = await res.json();

        if (history && history.length > 0) {
            // Tunggu sedikit agar intro muncul duluan sebelum history
            setTimeout(() => {
                history.forEach(msg => {
                    const role = msg.role === 'user' ? 'user' : 'ai';
                    const content = msg.content || msg.text || "";
                    addMessage(content, role);
                });
            }, 300);
        }
    } catch (err) {
        console.error("Failed to load history:", err);
    }
}

function clearChat() {
    // Custom in-chatbox confirm modal (replaces native confirm()).
    showConfirmModal(
        "Hapus semua chat?",
        "Riwayat percakapan ini akan dihapus dan tidak bisa dikembalikan.",
        doClearChat
    );
}

// In-chatbox confirm modal. Overlay is appended INTO #chat-box, which is
// position:fixed + overflow:hidden, so it's clipped to the chat window (not the
// whole page). Buttons resolve by calling onConfirm or just closing.
function showConfirmModal(title, body, onConfirm) {
    if (document.getElementById("ava-modal")) return;
    const overlay = document.createElement("div");
    overlay.id = "ava-modal";
    overlay.className = "ava-modal-overlay";
    overlay.innerHTML = `
        <div class="ava-modal-card">
            <h3 class="ava-modal-title">${title}</h3>
            <p class="ava-modal-body">${body}</p>
            <div class="ava-modal-actions">
                <button class="ava-modal-btn ava-modal-cancel">Batal</button>
                <button class="ava-modal-btn ava-modal-confirm">Hapus</button>
            </div>
        </div>
    `;
    const close = () => overlay.remove();
    overlay.querySelector(".ava-modal-cancel").onclick = close;
    overlay.querySelector(".ava-modal-confirm").onclick = () => { close(); onConfirm(); };
    overlay.addEventListener("click", (e) => { if (e.target === overlay) close(); });
    chatBox.appendChild(overlay);
}

async function doClearChat() {
    const sessionId = getSessionId();
    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL) ? API_BASE_URL : "";
    const headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    };

    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }

    try {
        await fetch(`${baseUrl}/api/v1/chat/history/${sessionId}`, {
            method: 'DELETE',
            headers: headers
        });
        messages.innerHTML = '';
        introduced = false;
        window._lastReflectiveQ = null;
        showIntro();
    } catch (e) {
        console.error("Error clearing chat history:", e);
    }
}

// ============================================================
// HELPERS
// ============================================================
function addMessage(text, type) {
    removeWelcome();
    const wrap = document.createElement("div");
    wrap.className = `msg ${type} animate__animated animate__zoomIn animate__faster`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${type}`;

    const formattedText = marked.parse(text);

    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = formattedText;
    tempDiv.querySelectorAll("a").forEach(link => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
    });

    bubble.innerHTML = `
        <div class="content">${tempDiv.innerHTML}</div>
    `;

    wrap.appendChild(bubble);
    messages.appendChild(wrap);
    messages.scrollTop = messages.scrollHeight;

    return wrap;
}

function showTyping() {
    const typingDiv = document.createElement("div");
    typingDiv.id = "typing-id";
    typingDiv.className = "msg ai animate__animated animate__fadeIn";
    typingDiv.innerHTML = `
        <div class="bubble ai">
            <div class="typing">
                <div class="dot"></div><div class="dot"></div><div class="dot"></div>
            </div>
        </div>
    `;
    messages.appendChild(typingDiv);
    messages.scrollTop = messages.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById("typing-id");
    if (el) el.remove();
}

// Configure marked for safe, clean rendering
marked.setOptions({ breaks: true, gfm: true });

// ============================================================
// SESSION ID — pakai Moodle User ID kalau tersedia
// ============================================================
function getSessionId() {
    if (typeof MOODLE_USER_ID !== 'undefined' && MOODLE_USER_ID > 0) {
        // MUST match $user_id_readable in block_chatbot.php byte-for-byte:
        //   {user_id}_{firstname}_{department}_{point}   e.g. "11_rossy_academy_ho"
        // Slug rule mirrors PHP: spaces→underscore, lowercase, then fallback.
        const slug = (v, fallback) => {
            const s = (typeof v === 'string' && v)
                ? v.replace(/\s+/g, '_').toLowerCase()
                : '';
            return s || fallback;
        };
        const nama = slug(typeof MOODLE_USER_NAME !== 'undefined' ? MOODLE_USER_NAME : '', 'user');
        const dept = slug(typeof MOODLE_DEPT !== 'undefined' ? MOODLE_DEPT : '', 'general');
        const point = slug(typeof MOODLE_POINT !== 'undefined' ? MOODLE_POINT : '', 'na');
        return `${MOODLE_USER_ID}_${nama}_${dept}_${point}`;
    }
    let sid = sessionStorage.getItem("ava_sid");
    if (!sid) {
        sid = "sid-" + Math.random().toString(36).substring(2, 9);
        sessionStorage.setItem("ava_sid", sid);
    }
    return sid;
}

function resetChat() {
    sessionStorage.removeItem("ava_sid");
    window.location.reload();
}

// ============================================================
// STREAMING BUBBLE HELPERS
// ============================================================

/**
 * Create an empty AI message bubble ready for streaming tokens into.
 * Returns { wrap, contentDiv, bubble } to allow progressive updates.
 */
function createStreamBubble() {
    removeWelcome();
    let wrap = document.getElementById("typing-id");
    let bubble;

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";

    if (wrap) {
        // Reuse existing bubble to prevent visual drop
        wrap.removeAttribute("id");
        bubble = wrap.querySelector(".bubble");
        if (bubble) {
            bubble.classList.add("streaming");
            const typingDiv = bubble.querySelector(".typing");
            if (typingDiv) typingDiv.remove();

            bubble.appendChild(contentDiv);
        }
    } else {
        wrap = document.createElement("div");
        wrap.className = "msg ai";

        bubble = document.createElement("div");
        bubble.className = "bubble ai streaming";

        bubble.appendChild(contentDiv);
        wrap.appendChild(bubble);
        messages.appendChild(wrap);
    }

    messages.scrollTop = messages.scrollHeight;

    return { wrap, contentDiv, bubble };
}

/**
 * Finalize the streaming bubble: remove cursor, remove streaming class.
 */
function finalizeStreamBubble(contentDiv, bubble, fullText) {
    bubble.classList.remove("streaming");

    // Final accurate render
    const tempDiv = document.createElement("div");
    tempDiv.innerHTML = marked.parse(fullText);
    tempDiv.querySelectorAll("a").forEach(link => {
        link.setAttribute("target", "_blank");
        link.setAttribute("rel", "noopener noreferrer");
    });
    contentDiv.innerHTML = tempDiv.innerHTML;
    messages.scrollTop = messages.scrollHeight;
}

// ============================================================
// SEND MESSAGE — True SSE Streaming
// ============================================================
async function send(presetText, opts) {
    // presetText: send this instead of the textarea value (used by the
    // "Ya, pandu aku" accept handler to re-ask the user's last question in
    // mentoring mode). opts.skipBubble: don't add a user bubble (the question
    // is already shown above, so re-asking shouldn't duplicate it).
    opts = opts || {};
    const text = (presetText != null ? presetText : textarea.value).trim();
    if (!text || isStreaming) return;

    removeCoachOffer();
    if (!opts.skipBubble) {
        addMessage(text, "user");
    }
    if (presetText == null) {
        textarea.value = "";
        textarea.style.height = "auto";
    }

    // Trigger #3 (explicit): user literally asked to be mentored while the mode
    // is OFF — show the offer instead of answering in normal mode. skipBubble is
    // the "Ya, pandu aku" re-ask path; never intercept it or we'd loop. The user
    // request already shows as a bubble above; the offer card appears below it.
    if (!window.COACHING_MODE && !opts.skipBubble && _wantsCoaching(text)) {
        const topic = _stripCoachingPhrase(text);
        window._lastReflectiveQ = topic || null;
        offerExplicitCoaching();
        return;
    }

    showTyping();
    isStreaming = true;
    currentAbortController = new AbortController();
    setSendButtonState(true);
    let streamWrap = null;

    const baseUrl = (typeof API_BASE_URL !== 'undefined' && API_BASE_URL)
        ? API_BASE_URL
        : "";

    const headers = {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "true"
    };

    if (typeof MOODLE_JWT !== 'undefined' && MOODLE_JWT) {
        headers["Authorization"] = `Bearer ${MOODLE_JWT}`;
    }

    const body = JSON.stringify({
        query: text,
        conversation_id: getSessionId(),
        course_id: typeof MOODLE_COURSE_ID !== 'undefined' ? MOODLE_COURSE_ID : 0,
        course_name: typeof MOODLE_COURSE_NAME !== 'undefined' ? MOODLE_COURSE_NAME : '',
        coaching_mode: !!window.COACHING_MODE
    });

    try {
        const res = await fetch(`${baseUrl}/api/v1/chat/stream`, {
            method: "POST",
            headers: headers,
            body: body,
            signal: currentAbortController.signal
        });

        if (!res.ok) {
            if (res.status === 429) {
                throw new Error("RATE_LIMIT");
            }
            throw new Error(`Server returned ${res.status}`);
        }

        // ── Init SSE reader ──
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        let contentDiv = null;
        let bubble = null;
        let _streamStarted = false;
        let _targetText = "";
        let _displayedText = "";
        let _streamActive = true;
        let _finalized = false;
        let _suggestCoaching = null;  // backend auto-hook signal (set in done event)
        let _coachingTopic = null;     // topic name when the offer was streak-triggered
        let _coachingDone = false;     // backend signal: Socratic loop wrapped up (set in done event)

        function startStreamBubble() {
            if (!_streamStarted) {
                const bubbleObj = createStreamBubble();
                streamWrap = bubbleObj.wrap;
                contentDiv = bubbleObj.contentDiv;
                bubble = bubbleObj.bubble;
                _streamStarted = true;
            }
        }

        function smoothStreamWorker() {
            if (!_streamStarted) {
                if (_streamActive) setTimeout(smoothStreamWorker, 20);
                return;
            }

            if (_displayedText.length < _targetText.length) {
                const remaining = _targetText.length - _displayedText.length;
                // Reveal chunk by chunk to catch up smoothly
                const chunkSize = Math.max(1, Math.floor(remaining / 4));
                _displayedText += _targetText.substring(_displayedText.length, _displayedText.length + chunkSize);

                // Add inline cursor BEFORE parsing markdown so it stays inside paragraph blocks <p>
                const renderText = _displayedText + '<span class="streaming-cursor-inline">▍</span>';

                const tempDiv = document.createElement("div");
                tempDiv.innerHTML = marked.parse(renderText);
                tempDiv.querySelectorAll("a").forEach(link => {
                    link.setAttribute("target", "_blank");
                    link.setAttribute("rel", "noopener noreferrer");
                });
                contentDiv.innerHTML = tempDiv.innerHTML;
                messages.scrollTop = messages.scrollHeight;
            }

            if (_streamActive || _displayedText.length < _targetText.length) {
                setTimeout(smoothStreamWorker, 20); // Fast but smooth 20ms frame
            } else if (!_finalized) {
                _finalized = true;
                const finalText = _targetText || "Hmm, jawabanku barusan nggak kekirim nih, kayaknya ada gangguan sebentar. Coba ketik ulang pertanyaannya dengan kalimat yang agak beda ya 🙏";
                finalizeStreamBubble(contentDiv, bubble, finalText);
                // Auto-hook: after the answer lands, offer coaching. Backend
                // signal (_suggestCoaching) is authoritative when present
                // (semantic affinity OR topic-streak); regex _looksReflective is
                // the fallback when it's absent. _coachingTopic words the offer
                // around the streak topic when the streak triggered it.
                maybeOfferCoaching(text, _suggestCoaching, _coachingTopic);
                maybeOfferBackToMentoring(_coachingDone);
                // Pending-response popup: if the chatbox was closed BEFORE the
                // answer finished streaming, show the answer as a popup above
                // the FAB. This is the "user walked away while Ava was typing"
                // case — they come back, FAB is pulsing fast, popup shows the
                // answer preview, click → opens chatbox to full answer.
                // Skip the empty-answer fallback (no point popping up an error).
                if (_targetText && _isChatboxHidden()) {
                    showPendingPopup(finalText);
                }
            }
        }

        smoothStreamWorker();

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            const lines = buffer.split("\n");
            buffer = lines.pop() || "";

            let currentEventType = "";

            for (const line of lines) {
                if (line.startsWith("event: ")) {
                    currentEventType = line.substring(7).trim();
                    continue;
                }

                if (line.startsWith("data: ")) {
                    const jsonStr = line.substring(6);

                    try {
                        const parsed = JSON.parse(jsonStr);

                        if (currentEventType === "resolved") {
                            // Biarkan UI natural, tidak perlu me-replace chat user dengan rewrite dari AI
                            currentEventType = "";
                            continue;
                        }

                        if (currentEventType === "done") {
                            startStreamBubble();
                            if (parsed.suggest_coaching !== undefined) _suggestCoaching = parsed.suggest_coaching;
                            if (parsed.coaching_topic !== undefined) _coachingTopic = parsed.coaching_topic;
                            if (parsed.coaching_done !== undefined) _coachingDone = parsed.coaching_done;
                            _streamActive = false;
                            currentEventType = "";
                            continue;
                        }

                        if (currentEventType === "error") {
                            startStreamBubble();
                            _targetText = "⚠️ Waduh, ada masalah saat memproses pertanyaan kamu. Coba lagi ya!";
                            _streamActive = false;
                            currentEventType = "";
                            continue;
                        }

                        if (parsed.token !== undefined) {
                            startStreamBubble();
                            _targetText += parsed.token;
                        }

                    } catch (parseErr) {
                        console.warn("SSE parse error:", parseErr, jsonStr);
                    }

                    currentEventType = "";
                    continue;
                }

                if (line.trim() === "") {
                    currentEventType = "";
                }
            }
        }

        // Signal stream logic is finished receiving
        _streamActive = false;



    } catch (err) {
        if (err.name === 'AbortError') {
            console.log("Request cancelled by user.");
            removeTyping();
            return;
        }
        console.error("Stream Error:", err);

        if (err.message === "RATE_LIMIT") {
            removeTyping();
            addMessage("⏳ Sabar dulu ya! Kamu udah ngelebihin batas 20 chat per menit. Tunggu sebentar lagi baru tanya lagi. 😊", "ai");
            return;
        }

        // ── Fallback: try non-streaming endpoint ──
        // Jangan removeTyping() dulu — biarkan dots tetap tampil selama fallback request
        // Kalau streaming sudah mulai (streamWrap ada), typing sudah digantikan bubble — tetap lanjut
        if (!streamWrap) {
            // Pastikan typing dots tetap ada / tampilkan ulang untuk fallback
            if (!document.getElementById("typing-id")) showTyping();
        }

        try {
            console.log("Falling back to non-streaming /chat endpoint...");
            const fallbackRes = await fetch(`${baseUrl}/api/v1/chat`, {
                method: "POST",
                headers: headers,
                body: body,
                signal: currentAbortController ? currentAbortController.signal : undefined
            });

            if (!fallbackRes.ok) {
                if (fallbackRes.status === 429) {
                    throw new Error("RATE_LIMIT");
                }
                throw new Error(`Fallback returned ${fallbackRes.status}`);
            }

            const data = await fallbackRes.json();
            removeTyping(); // hapus dots SETELAH response tiba

            const reply = data?.answer || "Hmm, jawabanku barusan nggak kekirim nih, kayaknya ada gangguan sebentar. Coba ketik ulang pertanyaannya dengan kalimat yang agak beda ya 🙏";
            streamMessageFallback(reply, "ai");
            maybeOfferCoaching(text);

        } catch (fallbackErr) {
            if (fallbackErr.name === 'AbortError') {
                console.log("Fallback request cancelled by user.");
                removeTyping();
                return;
            }
            console.error("Fallback also failed:", fallbackErr);
            removeTyping();
            if (fallbackErr.message === "RATE_LIMIT") {
                addMessage("⏳ Sabar dulu ya! Kamu udah ngelebihin batas 20 chat per menit. Tunggu sebentar lagi baru tanya lagi. 😊", "ai");
            } else {
                addMessage("⚠️ Waduh, koneksi ke server lagi bermasalah nih. Coba lagi ya!", "ai");
            }
        }
    } finally {
        isStreaming = false;
        setSendButtonState(false);
        currentAbortController = null;
    }
}

// ============================================================
// FALLBACK: Client-side simulated streaming (when SSE fails)
// ============================================================
function streamMessageFallback(fullText, type) {
    removeWelcome();
    const wrap = document.createElement("div");
    wrap.className = `msg ${type}`;

    const bubble = document.createElement("div");
    bubble.className = `bubble ${type}`;

    const contentDiv = document.createElement("div");
    contentDiv.className = "content";

    bubble.appendChild(contentDiv);
    wrap.appendChild(bubble);
    messages.appendChild(wrap);

    let currentText = "";
    let i = 0;

    const speed = Math.max(10, Math.min(30, 2000 / fullText.length));

    function typeChar() {
        if (i < fullText.length) {
            const chunkSize = Math.max(1, Math.floor(fullText.length / 100));
            currentText += fullText.substring(i, i + chunkSize);
            i += chunkSize;

            const renderText = currentText + '<span class="streaming-cursor-inline">▍</span>';
            const tempDiv = document.createElement("div");
            tempDiv.innerHTML = marked.parse(renderText);
            tempDiv.querySelectorAll("a").forEach(link => {
                link.setAttribute("target", "_blank");
                link.setAttribute("rel", "noopener noreferrer");
            });
            contentDiv.innerHTML = tempDiv.innerHTML;
            messages.scrollTop = messages.scrollHeight;
            setTimeout(typeChar, speed);
        } else {
            finalizeStreamBubble(contentDiv, bubble, fullText);
        }
    }

    typeChar();
    return wrap;
}

function addAIResponse(text) {
    streamMessageFallback(text, "ai");
}

function handleKey(e) {
    if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        if (!isStreaming) {
            handleSendClick();
        }
    }
}

// Auto resize textarea
const promptNode = document.getElementById("prompt");
if (promptNode) {
    promptNode.addEventListener("input", function () {
        this.style.height = "auto";
        this.style.height = (this.scrollHeight) + "px";
    });
}


