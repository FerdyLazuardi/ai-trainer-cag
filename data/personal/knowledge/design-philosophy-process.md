# My Design Philosophy & Process

This file answers questions about how I approach a new project, my design process, what makes me different, working with SMEs, AI in learning design, design philosophy, working approach, decision-making, prinsip kerja saya.

## Core philosophy in one sentence

**Outcome-first, evidence-driven, brutally concrete.** Every module starts with "what behavior changes?", every design choice ties back to that, and every claim is backed by a number — N-Gain, completion rate, or learner satisfaction.

## What this looks like in practice

### 1. I refuse to start with content
SMEs almost always start with "here's all the content you need to put in a course." I push back: "what does the learner DO differently after this?" If we can't answer that in one sentence, the course doesn't get built — we recommend a job aid, a process change, or nothing at all.

### 2. I bridge instructional design and engineering
Most Learning Designers stop at Articulate Storyline or Rise 360. I go further — RAG-based AI assistants (LangGraph + LlamaIndex + Qdrant) integrated directly into Moodle LMS. **I can write a learning objective in the morning and debug a FastAPI endpoint in the afternoon.** This is rare and it's deliberately my career bet.

### 3. AI changes what I build, not what I believe
AI is a *delivery channel* for learning, not a replacement for instructional design discipline. The Amartha LMS chatbot I'm building still relies on:
- Curriculum mapping (so the AI knows which content matters)
- Bloom's-aligned objectives (so retrieval surfaces the right cognitive level)
- Pre/post N-Gain to validate that AI-assisted learning actually outperforms text-only

### 4. I work in vertical slices
Build one full module end-to-end (script → animation → assessment → SCORM export → LMS deploy → pilot) before scaling. Modal Cycle Zero shipped one module to ~50 FOs first; once we hit target N-Gain, we scaled to 4,110.

## How I work with SMEs

* **Storyboard before script.** Visual-first review catches misalignment cheaper than reading paragraphs.
* **Decisions on the call, not in email.** Async review for SME edits OK; design decisions need synchronous time.
* **Document the "no's".** When a SME wants to add content I think violates Coherence/cognitive-load, I document the trade-off and let them decide. They almost always self-correct after seeing it written.

## How I handle conflicting feedback

Conflict between stakeholders is normal. My playbook:

1. **Re-ground in objectives.** If two stakeholders disagree on content, both are usually drifting from the original learner-outcome.
2. **Test, don't argue.** Pilot two versions on a 20-person sample.
3. **Escalate by data.** N-Gain numbers settle most arguments faster than opinion.

## What makes my work measurable

Every project I've shipped has at least L1 (satisfaction) + L2 (N-Gain) data:
- **Modal Cycle Zero** — N-Gain **44.63%**, **4,110 participants**
- **Anti Harassment** — N-Gain **64.58%**, completion **75.6%**
- **TCP** — N-Gain **44.64%**, scores **54.04 → 79.12**
- **Dunia Geometri** — N-Gain **0.57** (moderate), expert validity **91.6%**

I can't always claim L3/L4 (behavior change, business KPIs) without sponsor buy-in — but L1+L2 is non-negotiable on every course I touch.

## Working style

* **Honest about constraints.** Will tell you up front "this content doesn't need a course" rather than build padding.
* **Moves between tools fluently.** Storyline for branching, Rise for fast turnarounds, Articulate Review 360 for SME feedback, Figma for assets, Python+LangGraph when off-the-shelf isn't enough.
* **Comfortable with ambiguity early, allergic to ambiguity late.** Day-1 spec is fine to be vague. Pre-production storyboard cannot be.
