# StrideSight

AI-powered, on-device sprint biomechanics analysis — no video ever leaves the browser.

![StrideSight screenshot](docs/screenshot.png)

## What it does

Upload a sprint video (MP4, MOV, or WebM) and StrideSight runs Google's MediaPipe Pose model entirely client-side to extract frame-by-frame joint kinematics — trunk lean, knee drive, hip extension, and arm swing — each computed from 2D vector dot products between tracked landmarks.

The core idea: acceleration and max-velocity sprinting are biomechanically different gaits, not the same gait done "better" or "worse." A sprinter's forward trunk lean — the signal sports science literature uses to distinguish these phases — is tracked in real time and used to classify every frame into **Acceleration**, **Transition**, or **Max Velocity**. Each phase is then scored against its own literature-informed targets, rather than judging an entire sprint against one fixed standard.

## Features

- **Phase-aware scoring** — separate 0–100 form scores per detected sprint phase, each grounded in phase-specific biomechanical targets (see [Research grounding](#research-grounding)).
- **Real-time skeleton + joint-angle overlay** — a canvas layer draws the tracked skeleton and semi-transparent arcs at each measured joint, color-coded to how that angle compares to the current phase's targets.
- **Coaching feedback, not just numbers** — specific, numbers-backed recommendations *and* strengths ("What's Working"), a highlighted top-priority focus area, and a synthesized cross-phase narrative summary.
- **Step cadence** — causal peak-detection on the knee-drive signal estimates step frequency, reported as context rather than graded — published research shows elite sprinters are legitimately frequency-reliant *or* length-reliant, so there's no single "correct" cadence to impose.
- **Symmetry checks** — flags persistent left/right gaps in arm swing or knee drive that can indicate a strength imbalance or compensation pattern.
- **Manual phase override** — auto-detection reads trunk lean and needs a roughly side-on camera angle; an override lets you force a phase if your footage makes auto-detection unreliable.
- **CSV export** — full frame-by-frame data plus per-phase summary blocks (scores, recommendations, strengths), ready for external analysis.
- **100% client-side** — video is decoded and analyzed frame-by-frame in the browser and never uploaded anywhere.

## Tech stack

- **Next.js 14** (App Router) + **TypeScript**
- **Tailwind CSS**
- **MediaPipe Pose** (dynamically imported client-side — see [Architecture notes](#architecture-notes))
- **lucide-react** for icons

## Getting started

```bash
git clone https://github.com/alexandrio1710/stridesight.git
cd stridesight
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and drop in a sprint video.

## Project structure

```
app/
  layout.tsx         Root layout
  page.tsx           Landing page / header
  globals.css        Tailwind entrypoint
components/
  VideoAnalyzer.tsx  Client component: video/canvas, MediaPipe lifecycle, dashboard UI
utils/
  biomechanics.ts    Pure math/scoring engine — no DOM, no React, fully unit-testable
```

`utils/biomechanics.ts` is deliberately framework-agnostic: vector math, phase classification, continuous scoring, and recommendation/strength generation all live there as pure functions operating on plain data, independent of MediaPipe or React.

## How the biomechanics engine works

1. **Landmark → angle.** Every tracked angle (trunk lean, knee drive, hip extension, arm swing) is computed via the same vector dot-product identity, `cos(θ) = (BA · BC) / (|BA| |BC|)`, evaluated at the relevant joint.
2. **Phase classification.** Trunk lean is smoothed with an exponential moving average and classified into a sprint phase using a Schmitt-trigger-style hysteresis band, so the detected phase doesn't flicker if lean is hovering near a boundary.
3. **Phase-specific thresholds.** Each phase has its own optimal/caution bands per metric — e.g. max-velocity's hip-extension target is intentionally *less* extreme than acceleration's, because published research shows elite sprinters terminate ground contact before reaching full extension at top speed, favoring fast leg recovery over maximal extension.
4. **Continuous scoring.** A 0–100 score per metric is derived from the same threshold constants used for the live 3-color status, via a continuous piecewise-linear mapping (not a discontinuous step).
5. **Recommendations & strengths.** Every metric below its optimal threshold generates a specific, numbers-backed recommendation; every metric that clears it generates an equally specific strength callout. A cross-phase narrative then synthesizes across all reliably-scored phases into a short coaching summary.

## Research grounding

Phase-specific thresholds are synthesized from published sprint biomechanics literature, not invented numbers or a trained model — MediaPipe's pose detector is the only pretrained model in this pipeline; everything downstream is a rules-based coaching heuristic:

- [Transition from upright to greater forward-lean posture predicts faster acceleration](https://www.sciencedirect.com/science/article/pii/S0966636223013085)
- [Sprint start biomechanics](https://auptimo.com/sprint-start-biomechanics/)
- [Kinematic stride characteristics of maximal sprint running of elite sprinters](https://pmc.ncbi.nlm.nih.gov/articles/PMC8008308/)
- [Angular kinematics during top-speed sprinting](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11994691/)
- [Sprint mechanics: phases, technique, and speed cues](https://thecompleteathleteguide.com/sprint-mechanics/)
- [Biomechanics of sprint running (overview)](https://en.wikipedia.org/wiki/Biomechanics_of_sprint_running)

## Limitations

- **Camera angle matters.** Trunk lean — the phase-detection signal — is only meaningful from a roughly side-on (lateral) view. Head-on or behind-the-runner footage will give unreliable phase classification; use the manual override in that case.
- **These are coaching heuristics, not individual prescriptions.** Elite sprinters vary meaningfully in technique by body type, event, and training background. Scores are a conversation-starter, not a verdict — step cadence in particular is reported without a "correct" number, since research shows elite sprinters are legitimately frequency-reliant or length-reliant.
- **2D pose only.** MediaPipe Pose estimates 2D landmarks; there's no camera calibration, so no true velocity, distance, or 3D angle is measured — only image-plane angles.

## Architecture notes

`@mediapipe/pose` touches `window`/`self` at module-evaluation time, which breaks Next.js's server-side render pass if statically imported — even inside a `'use client'` component. It's loaded via a dynamic `import()` inside a `useEffect` instead, guaranteeing it only ever executes client-side.
