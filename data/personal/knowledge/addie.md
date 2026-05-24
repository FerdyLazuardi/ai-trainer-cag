# How I apply ADDIE

## Quick answer

I use **ADDIE** (Analyze → Design → Develop → Implement → Evaluate) as a backbone for any non-trivial training project, but I run it iteratively rather than waterfall-style. Each phase produces a concrete artifact that the next phase consumes — so SMEs and stakeholders can sign off step-by-step instead of waiting until launch.

## How each phase looks in practice

### A — Analyze
Before opening Storyline or writing a single script, I map:
- **Audience** — who they are, what they already know, the conditions they'll learn under (Field Officers learning on a phone in the field is very different from head-office staff at a desk).
- **Performance gap** — what they should be doing on the job that they aren't doing yet, and why. I push back on training-as-default — sometimes the answer is a job aid or a process change, not a course.
- **Constraints** — bandwidth, devices, language, time available per session, regulatory requirements.

For **Modal Cycle Zero**, the Analyze phase surfaced that FOs needed not just product knowledge but also **objection-handling under pressure** — that re-shaped the whole curriculum away from a feature-list approach toward simulation-heavy practice.

### D — Design
This is where Bloom's Taxonomy and curriculum mapping live. Output of Design is:
- A **curriculum map** — competencies → objectives → topics → assessment items, all aligned.
- **Storyboards** for each module — scene-by-scene plan with narration, visuals, and interaction type.
- **Assessment blueprint** — pre-test, post-test, and N-Gain target before any content is built.

I split scripts into **Lecture scripts** (didactic, encouraging tone — "Hai A-Team!") and **Simulation scripts** (relatable characters, conflict-resolution arcs) so each segment hits its intended cognitive level.

### D — Develop
Production phase: build modules in **Articulate Storyline / Rise**, animate in After Effects / Capcut, design assets in Figma / CorelDraw. For interactive AI components I use **Python, LangGraph, LlamaIndex, Qdrant** (the same stack that powers this Askfer assistant).

I prototype early and have the SME sanity-check storyboards before full production, so we don't burn animation hours on something that needs rewriting.

### I — Implement
For Amartha projects this means deploying to the Moodle LMS as SCORM packages, with rollout coordination across regions. I make sure FO/BP teams have device-tested modules and that the kickoff comms are clear about what's being learned and why.

### E — Evaluate
This is where most "ADDIE in name only" projects fail — they skip evaluation because launching feels like the finish line. I bake evaluation into Design from day one:
- **Reaction (Kirkpatrick L1)** — quick post-module survey on relevance and clarity.
- **Learning (L2)** — pre-test / post-test pairs with **N-Gain** as the headline metric.
- **Behavior (L3)** — where possible, follow up with managers/coaches on whether on-the-job behavior changed.

Concrete results from my projects:
- **Modal Cycle Zero** — N-Gain **44.63 % (Medium Gain)** across the field team, validated via paired pre/post.
- **Training Client Protection** — average score moved from **54.04 (pre) → 79.12 (post)**, N-Gain **44.64 % (Medium Gain)**.
- **Dunia Geometri** — math achievement scores moved from **46 → 76** (+30 points), with **84.6 %** measured learning interest.

The Evaluate phase data feeds back into Analyze for the next cycle — that's the iteration loop most teams never close.

## How I run ADDIE in practice — not waterfall

Pure-waterfall ADDIE is a trap; real projects need feedback loops. My version:
- Analyze and Design overlap: I prototype objectives + storyboards in parallel and refine both as I learn.
- Develop runs in **vertical slices**: build one full module end-to-end (script → animation → quiz) before scaling up, so quality issues surface fast.
- Evaluate happens **mid-project** too — I'll pilot one module to a sample group before producing the rest of the curriculum.

## When I deviate from ADDIE

For tight-timeline, high-iteration work (e.g. internal tools, AI prototypes) I lean toward **SAM (Successive Approximation Model)** — three-phase rapid iteration with frequent prototypes — and use ADDIE artifacts (curriculum map, assessment blueprint) as deliverables inside SAM cycles rather than as gates.

## Why I use it

ADDIE isn't fashionable, but it's rigorous and shareable. SMEs, project managers, and recruiters all understand the vocabulary. It forces me to articulate **why** every module exists and **how** I'll know it worked — and that discipline shows up in the N-Gain numbers above.
