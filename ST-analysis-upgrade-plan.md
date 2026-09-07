# S-T Analysis Based Classroom Engagement — Upgrade Plan

## What is S-T Analysis?

S-T Analysis (Student-Teacher Analysis) is a classroom observation method widely used in Chinese educational technology research. Every N seconds, classroom behavior is coded into two categories:

| Code | Meaning | Student-side equivalent |
|---|---|---|
| **T** | Teacher-active (lecturing, questioning, demonstrating) | **Student-passive** — listening, looking forward, still |
| **S** | Student-active (answering, discussing, presenting, peer review) | **Student-active** — talking, writing, hand raised, peer interaction |

For ReLi, we adapt this to **per-student** S/T classification and aggregate to class-level metrics.

## Why S-T Instead of the Current Approach

| Aspect | Current (50/30/20 formula) | S-T Analysis |
|---|---|---|
| Student states | Continuous score (0-100) | Binary: S (active) or T (passive) |
| Behavioral scope | Gaze + 8 emotions + yawn | Talking, writing, hand-raising, peer interaction, still, looking away |
| Research basis | Arbitrary weights | Established in Chinese classroom research (Fujita, 1987; widely adapted in China) |
| Teacher insight | "Engagement: 68%" — opaque | "18 of 30 students are active" — actionable |
| Activities detected | 3 limited signals | Full range of observable classroom behaviors |
| Model complexity | 3 separate models (Haar + MediaPipe + 2 ONNX) | Single YOLO-based behavioral classifier |
| Occlusion handling | None | IMPDIoU loss (from Xiao et al. 2024) |

## Overview of Changes

```
Current Pipeline:
  Camera → Haar Cascade → MediaPipe → ONNX Gaze → 50/30/20 Score
                              ↓                          ↓
                         ONNX Emotion              Single % number
                              ↓
                         MediaPipe Yawn

New Pipeline (S-T):
  Camera → YOLOv8n (detects S/T behaviors per student) → S/T per face
                           ↓
                    Aggregation → %S, %T, Transition Rate, Activity Log
                           ↓
                    Frontend: S/T ratio chart + activity heatmap
```

## Behavior-to-Class Mapping

| S-Class (Student Active) | T-Class (Student Passive) |
|---|---|
| Talking / mouth moving | Looking forward / at board |
| Hand raised | Head down (reading/writing silently) |
| Writing / note-taking | Still / resting |
| Turned to peer (discussion) | Eyes closed / resting |
| Standing / presenting | On phone (disengaged variant) |
| Peer review / group work | Drinking water |
| Applauding | |

Note: Some T-class behaviors have sub-variants (engaged-passive vs disengaged-passive). The initial model treats all non-S behavior as T. A future refinement can split T into T-engaged and T-disengaged.

## Work Packages

### WP1 — Dataset

| Task | Detail |
|---|---|
| Record classrooms | Fixed camera above blackboard, 1920×1080, 2–3 sessions × 45–90 min |
| Annotate with LabelImg | Per-student bounding boxes + S/T label (2 classes, simpler than 5-class ICAPD) |
| Dataset size target | ~3,000–5,000 annotated frames (~86 students per the Xiao paper setup) |
| Data augmentation | Mosaic + Cutout (same as Xiao et al.) — helps with occlusion and small dataset |
| Class balance | Expect ~70% T, ~30% S in a typical lecture — use class-weighting to avoid bias |

### WP2 — Detection Model

| Component | Choice | Rationale |
|---|---|---|
| Base architecture | YOLOv8n | Real-time, lightweight, good occlusion handling baseline |
| Loss function | IMPDIoU (from Xiao et al.) | Handles occluded students in dense classrooms |
| Input resolution | 640×640 (YOLOv8 default) | Good balance of speed vs accuracy |
| Training epochs | 300 | Matches OE-YOLOv8n setup |
| Batch size | 128 (with RTX 3090) | As per paper |
| Output | Per-student: bbox + S/T class + confidence | |

**Success criteria**: mAP50 ≥ 0.90 on test set, real-time at ≥ 10 FPS on GPU.

### WP3 — S-T Metrics Computation

Replace `aggregate_class_metrics()` and `compute_engagement()` with:

```
Per-frame (every 3 seconds):
  - YOLOv8n detects all students + S/T class
  - Count: n_S, n_T, total students detected
  - Rt = n_T / total (teacher-ratio, aligned with S-T methodology)
  - Class activity status: "Hyperactive" if Rt < 0.3, "Balanced" if 0.3–0.7, "Passive" if Rt > 0.7

Rolling window (last 5 minutes):
  - Transition rate: how often students switch S↔T
  - Dominant mode: which S/T pattern the class is in
  - Individual tracking: per-student S/T timeline (memory-safe, face-IDs only within session)

Alerts:
  - Configurable: "Passive ratio > 80% for 5+ minutes"
  - Transition rate alerts: "Very low transitions — possible disengagement"
```

### WP4 — Dashboard Redesign

| Element | What to show |
|---|---|
| **S/T Ratio Gauge** | Big circular indicator: % S vs % T, color-coded (green=active, blue=passive) |
| **S/T Timeline** | Stacked area chart over session duration (green S / blue T bands) |
| **Activity Feed** | Text list: "3 students talking (S) · 2 writing (S) · 1 hand raised (S)" |
| **Transition Indicator** | Arrow showing trend: "Active ↑ 8% in last 5 min" or "Becoming passive ↓" |
| **Mode Badge** | "Lecture Mode" (high T, low transitions) vs "Discussion Mode" (high S) vs "Group Work" (mixed) |
| **Alert Controls** | Teacher-adjustable: "Alert me when passive > __% for __ minutes" |

**Emotional tone**: Focus on **actionability** — the teacher should instantly know "how many students are engaged right now" and "is the trend improving or worsening?" rather than a single opaque percentage.

### WP5 — Real Classroom Testing

| Test | What to check |
|---|---|
| **Day 1 — Controlled** | 5–8 volunteers with known behaviors (read, talk, write, look away on cue) |
| **Day 2 — Live lecture** | Real 45-min class, teacher logs subjective S/T ratio every 5 min |
| **Day 3 — Peer review** | Group discussion session — validate S-detection during peer interaction |
| **Occlusion test** | Deliberately create crowding to test IMPDIoU |
| **Lighting test** | Dim classroom, side lighting, projector washout |

**Validation metrics**:
- S/T ratio vs teacher's subjective rating: within ±10%
- Transition count: matches observed behavior transitions
- Per-student accuracy ≥ 85% (sampled spot checks)

### WP6 — Test Infrastructure

| Layer | Tests |
|---|---|
| Unit | S/T aggregation logic, edge cases (0 students, all S, all T) |
| Model | YOLOv8n inference on synthetic frames, IMPDIoU loss correctness |
| Integration | Full pipeline: synthetic video → detection → S/T metrics → frontend format |
| E2E | Recorded classroom video → pipeline → verify dashboard matches manual annotation |
| Performance | FPS benchmark on GPU and CPU, memory usage with 35 students |

## Timeline

| Week | Work |
|---|---|
| 1 | Camera setup + record 2–3 classroom sessions |
| 2 | Annotate S/T labels with LabelImg (~3,000 frames) |
| 3 | Train YOLOv8n + IMPDIoU on collected data |
| 4 | Model evaluation + refinement (iterate on false positives/negatives) |
| 5 | Pipeline integration: replace old FrameDistributor, write S/T metrics |
| 6 | Test infrastructure + basic unit/integration tests |
| 7 | Real classroom testing (controlled → live → peer review) |
| 8 | Dashboard redesign + final polish |

## Comparison to Current Architecture

| Current file/component | Replace with |
|---|---|
| `engagement_engine.py` | `st_analysis.py` — S/T computation, transition rate, mode classification |
| `frame_distributor.py` | `yolo_detector.py` — single YOLOv8n inference (no more cascade/MediaPipe chain) |
| `gaze_onnx.py` | Remove — no longer needed |
| `emotion_onnx.py` | Remove — no longer needed |
| `yawn_detector.py` | Remove — yawn is one of many S/T indicators now |
| `FaceEngagementState` | `StudentActivityState` — S/T class, bbox, confidence, transition count |
| `aggregate_class_metrics()` | `aggregate_st_metrics()` — S/T ratio, transition rate, activity counts |
| Dashboard engagement chart | S/T stacked area chart + activity feed |
| Alert (40%) | Configurable alert on S/T ratio thresholds |

## Key Risks

| Risk | Mitigation |
|---|---|
| S-class data is rare in lectures (~30%) | Class-weighting in loss, Mosaic augmentation, collect discussion-session footage |
| YOLOv8n real-time on classroom camera | Use YOLOv8n (nano) — ~15 FPS on RTX 3090 at 640×640; fallback to every-3s sampling |
| Students have varying definitions of "active" | Consistent annotation guide for labelers (define clear S/T boundary) |
| Frontend needs to show per-student data without privacy leaks | Only show aggregated S/T ratio + anonymized activity feed (no face images or identities) |
