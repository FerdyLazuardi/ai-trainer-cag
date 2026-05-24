# How I Measure Impact in E-Learning Projects

This file answers questions about measuring training impact, e-learning ROI, evaluating effectiveness, learning analytics, KPIs for L&D, gimana ngeliat impact e-learning, cara ngukur dampak training, KPI learning designer, learning analytics dashboard, beyond completion rate.

## Quick answer

Completion rate alone is **not impact**. As a Learning Designer for e-learning specifically, I track impact across **5 dimensions**, ordered from cheapest to hardest to capture:

1. **Engagement signals** (clickstream, time-on-task, drop-off curves)
2. **Reaction** (Kirkpatrick L1 — satisfaction surveys)
3. **Learning** (Kirkpatrick L2 — N-Gain via pre/post)
4. **Behavior** (Kirkpatrick L3 — applied practice on the job)
5. **Business outcome** (Kirkpatrick L4 — KPIs the sponsor actually cares about)

Most L&D teams stop at L1. I commit to L1+L2 minimum on every project, and propose L3/L4 wherever a sponsor is willing.

## Concrete data I capture per project

### Engagement signals (free, from any LMS)
* **Module-by-module drop-off curve** — where do learners leave? That's the design problem.
* **Time-on-task vs. estimated time** — way under = clicking-through, way over = struggling.
* **Quiz first-attempt success rate** — proxy for content clarity.
* **Re-entry rate** — do they come back to revisit?

For Anti-Harassment (1,980 participants), the drop-off curve was nearly flat — confirmed segmentation discipline worked.

### L1 — Reaction (post-course survey)
* Likert 1–4 across: relevance, clarity, applicability.
* Aggregate **3.64 / 4** across portfolio. I aim for ≥3.5 per individual course.
* Optional free-text: "what would you change?" — surfaces real engagement gaps.

### L2 — Learning (pre/post + N-Gain)
* Pre-test before module 1, post-test after final module. Same instrument, same difficulty.
* **Hake's normalized gain** — corrects for ceiling effects.
* My benchmarks: < 30% Low Gain (redesign), 30-70% Medium (acceptable), > 70% High (unusual, often instrument problem).
* **Anti Harassment N-Gain 64.58%**, **Modal Cycle Zero 44.63%**, **TCP 44.64%** — all Medium Gain on rigorous instruments.

### L3 — Behavior (when I can negotiate it)
* **Manager check-in survey** 30/60/90 days post-course: "have you seen [target behavior] from this person?"
* **Job aid usage analytics** — if learners actually pull up the post-course aid, learning transferred.
* **Recorded role-play / call review** — gold standard but expensive; reserved for high-stakes content (Modal Cycle Zero objection handling).
* **Self-reported applied practice** — quick + dirty; biased but useful as a leading indicator.

### L4 — Business outcome (sponsor-led)
* I propose **proxy KPIs at kickoff**, but the actual measurement is owned by the business.
* For Modal Cycle Zero: post-launch product disbursement velocity. For TCP: client-protection-related complaint volume.
* **L4 is rarely cleanly attributable to a single course** — I'm honest about this with stakeholders.

## My L&D dashboard (what I track per course)

| Metric | Why | Trigger to act |
|---|---|---|
| Completion rate | Headline | < 50% → redesign |
| Avg. time-on-task | Engagement | < 50% of estimate → check for click-through |
| Module drop-off curve | Design diagnosis | Cliff in any segment → fix that segment |
| L1 satisfaction | Sentiment | < 3.0 / 4.0 → free-text review |
| L2 N-Gain | Did they learn | < 30% → instrument or content problem |
| Quiz first-attempt rate | Clarity | < 60% on first try → content unclear |
| Re-entry rate | Stickiness | High = good for reference material, low = course is one-and-done |

## What I push back on

* **"Just give me NPS"** — NPS for training is noisy. Likert + free-text + N-Gain beats it.
* **"Completion = success"** — clicking-through completion is fake. Pair it with N-Gain.
* **"We don't have time for pre-test"** — then we can't claim impact. Either invest or stop calling it impact.
* **"Self-reported behavior change is enough"** — recall bias is real; needs a manager / observer signal at least.

## How I report impact to stakeholders

One slide, four numbers per course:

* **N participants** (audience reach)
* **Completion rate** (engagement)
* **N-Gain** (learning)
* **Satisfaction** (reception)

Plus 1 free-text quote that makes it human. If I can layer in L3 behavior change, I do. Stakeholders rarely need more than this — when they do, I drill into the dashboard above.

## Why I publish numbers on my portfolio

Recruiters get told "I'm a great learning designer" by everyone. Numbers force honesty: **Anti-Harassment** N-Gain **64.58%** with **1,980** participants is a defensible claim. "Made impactful courses" is not.
