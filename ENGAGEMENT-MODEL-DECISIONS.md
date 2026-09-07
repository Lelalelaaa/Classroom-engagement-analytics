# ReLi — Engagement Model Decisions One-Pager

Purpose: map every hardcoded constant in the engagement pipeline to its justification and evidence status, to drive the mentor discussion on survey design and model-choice evidence.

Legend: 🟢 literature-supported · 🟡 plausible but unverified · 🔴 arbitrary / chosen for convenience

---

## 1. Engagement score fusion

| # | Decision | Value | Where | Justification (current) | Evidence status |
|---|----------|-------|-------|-------------------------|-----------------|
| 1 | Signal weights (gaze/emotion/yawn) | 0.50 / 0.30 / 0.20 | `engagement_engine.py:20` | "Gaze is the strongest signal" — no source | 🔴 |
| 2 | Engaged threshold | ≥ 0.5 | `engagement_engine.py:26` | Midpoint convention | 🔴 |
| 3 | Gaze scored binary (0/1) | `1.0 if looking else 0.0` | `engagement_engine.py:57` | Simplicity; no partial credit for glancing | 🔴 |
| 4 | Yawn scored binary (0/1) | `0.0 if yawning else 1.0` | `engagement_engine.py:63` | Simplicity | 🟡 |
| 5 | Emotion → valence map | happy 1.0, surprised 0.8, neutral 0.6, sad 0.3, angry 0.2, fear 0.2, disgust 0.1 | `engagement_engine.py:9` | Ordering matches arousal/valence literature, but numeric values are hand-picked | 🟡 (ordering) / 🔴 (values) |
| 6 | Missing-emotion fallback | 0.6 (neutral) | `engagement_engine.py:44,61`; `frame_distributor.py:150` | Assume neutral when classifier unavailable | 🟡 |

**Note — duplicated/divergent constants:** a second valence map exists in `emotion_onnx.py:23` (surprised 0.75, contempt 0.4 instead of 0.8, 0.0) and the fusion formula is hardcoded again in `frame_distributor.py:153` (0.5/0.3/0.2) instead of importing `WEIGHTS`. The two maps disagree — risk of drift. Should be consolidated.

---

## 2. Model choices

| # | Model | Where | Why this one (current) | Evidence status |
|---|-------|-------|------------------------|-----------------|
| 7 | Face detection: OpenCV Haarcascade | `frame_distributor.py:52` | Fast, zero-download | 🟡 — known to be less accurate than DNN/MediaPipe detectors; rationale for speed vs. accuracy trade-off undocumented |
| 8 | Gaze: yakhyo/resnet18_gaze ONNX | `gaze_onnx.py` | Reference: GazeCapture paper lineage | 🟡 — GazeCapture is published (Krafka et al. 2016), but per-face accuracy on your camera setup unmeasured |
| 9 | Emotion: FER+ ONNX | `emotion_onnx.py` | Reference: onnx/models, FER+ paper lineage | 🟡 — FER+ has published benchmarks (Barsoum et al. 2016); accuracy on low-res classroom crops unmeasured |
| 10 | Landmarks: MediaPipe FaceLandmarker | `mediapipe_init.py` | Task API, auto-download, iris support | 🟢 — Google's published landmark model |

---

## 3. Signal thresholds & tunables

| # | Decision | Value | Where | Justification (current) | Evidence status |
|---|----------|-------|-------|-------------------------|-----------------|
| 11 | Gaze "centered" threshold | ±0.25 around ratio 0.5 | `gaze_estimator.py:22` | Hand-tuned | 🔴 |
| 12 | Gaze forward cone | \|yaw\|<15° and \|pitch\|<15° | `gaze_onnx.py:90` | Hand-tuned | 🔴 |
| 13 | Gaze ratio normalisation | ±30° maps to 0–1 | `gaze_onnx.py:87` | Assumes screen ≈ 60° wide at seated distance | 🟡 |
| 14 | Yawn MAR threshold | 0.6 | `yawn_detector.py:10` | Common range in MAR literature (0.5–0.6), but per-camera unverified | 🟢/🟡 |
| 15 | Emotion inference skip | every 5th frame | `emotion_onnx.py:42`, `emotion_classifier.py:16` | CPU budget | 🟡 |
| 16 | Face detector params | scaleFactor 1.1, minNeighbors 5, minSize 60×60 | `frame_distributor.py:97` | OpenCV defaults + hand-tuning | 🔴 |
| 17 | Max faces | 35 | `frame_distributor.py:85`, `mediapipe_init.py:65` | Typical classroom estimate | 🟡 |
| 18 | Landmarker confidence thresholds | 0.5 / 0.5 / 0.5 | `mediapipe_init.py:71-73` | Defaults | 🟡 |

---

## 4. Class-level & alerting constants

| # | Decision | Value | Where | Justification (current) | Evidence status |
|---|----------|-------|-------|-------------------------|-----------------|
| 19 | Class engagement = mean of per-face composites | — | `frame_distributor.py:211` | Simple aggregation | 🟡 — mean hides disengaged minorities; alternative: % under threshold |
| 20 | Low-engagement alert threshold | < 50.0 | `config.py:11` | Round number | 🔴 |
| 21 | Alert cooldown | 120 s | `config.py:12` | Arbitrary | 🔴 |
| 22 | Metric write interval | 30 s | `config.py:10` | Persistence convenience | 🟡 |

---

## 5. Open questions for the mentor

1. **Scope of the survey** — validate only the fusion weights (row 1), or also the valence map (row 5), thresholds (11–14, 20), and aggregation (19)? Recommend: at minimum weights + valence ordering.
2. **Who is the survey population?** Teachers (external observer), students (self-report), or both? These disagree in reality — if we only survey one, the weights may not generalise.
3. **How do raw answers become numbers?** Likert importance → normalise to sum 1.0? Or pairwise ranking → AHP? Pick before collecting.
4. **Where does model-choice evidence come from?** Papers (GazeCapture, FER+) are sufficient for the report, or do we need our own accuracy benchmark on classroom footage?
5. **How do we verify new weights are better?** Ground-truth labeled videos? Teacher ratings on the demo videos?
6. **Ethics** — is the survey exempt from ethics review (anonymous, adults) or does it need approval?

---

## 6. Immediate hygiene fixes (no research needed)

- Consolidate the duplicated valence maps and fusion weights into single constants imported everywhere.
- Make `WEIGHTS`, `EMOTION_VALENCE`, `ENGAGED_THRESHOLD` configurable via settings (env) so survey-derived values can be plugged in without code edits.
