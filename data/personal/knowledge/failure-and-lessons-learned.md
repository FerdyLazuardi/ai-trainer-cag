# Lessons from Things That Didn't Work

Every project I've shipped has had something that didn't go as planned. Here are the concrete cases — share these directly when asked about failures, projects that didn't work, didn't go as planned, mistakes, lessons learned, or what I'd redo.

### 1. TCP — completion gap surprised me

**Training Client Protection** had **2,031 participants** and **1,475 completed both pre/post tests** (**72.6%**) — but only **1,028 achieved full course completion** (**50.6%**) including the final certification step. That ~22 percentage-point gap was a design failure: the certification step had a friction issue (a feedback survey that wasn't adaptive on mobile). I didn't catch it pre-launch because I tested on desktop. **What I'd do differently:** mobile-first usability testing on the full learner flow, not just content modules. Now standard practice on every project.

### 2. Modal Cycle Zero — overestimated FO connectivity

Modal Cycle Zero reached **4,110 participants** with completion **54.8%** — lower than Anti Harassment's **75.6%**. Drop-off was geographically concentrated in regions with poor connectivity. **What I'd do differently:** designed an offline-first companion job aid (PDF + audio version) for low-connectivity regions earlier in the project, not as a post-launch patch. The structural connectivity issue was knowable at Analyze phase but I deprioritized it.

### 3. Early Dunia Geometri pilot — scope overflow

The first iteration tried to cover too much geometry in one module. Pre-pilot with teachers surfaced the issue: kids tuned out at minute 12. Re-scoped from "all 5th-grade geometry" to "introduction to flat shapes" — completion + interest jumped, but it cost me 3 weeks of rework. **What I'd do differently:** pilot earlier with smaller scope. Ship the minimum viable module, validate, then expand. Same lesson behind why I now work in vertical slices.

## Patterns I've learned to watch for

- **Stakeholder content-creep late in production** → unclear objectives. Push back to Analyze phase.
- **High satisfaction with low N-Gain** → assessment is too easy or mirrors content too closely. Redesign instrument.
- **Mobile vs desktop usability gap** → always test on the actual device the audience uses.
- **Pilot-skipping under deadline pressure** → every time I've shipped without piloting, I've paid for it post-launch.

## Areas I'm still developing

- **L3 behavior measurement** — capturing on-job behavior change rigorously.
- **Quasi-experimental designs** — control vs. treatment groups in corporate L&D.
- **Production-grade AI evaluation** — for AI-assisted learning, evaluation methodology is still evolving.

---

This file answers: projects that didn't work, didn't go as planned, failed, went wrong, mistakes, lessons learned, things I'd redesign, design failures, weakness, area of growth, kegagalan dalam project, hal yang aku belajar, what would you do differently, biggest mistake, hardest lesson, regret, reframe.
