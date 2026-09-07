# Offline evaluation & threshold tuning

Every threshold in the attention model (`app/config.py`, "Attention model" block)
is an assumption until it has been fitted against real footage. This directory is
how that fitting happens. Nothing here runs in the live server or touches the
live privacy boundary — it operates on lessons that were recorded out of band,
with consent, and are never committed.

## The two feedback sources

| Source | Who produces it | Granularity | Fitter |
|---|---|---|---|
| `annotations.csv` | a research annotator (plus a second for agreement) | per student, per sampled frame | `sweep.py` |
| `segment_ratings.csv` | the classroom teacher | one rating per 5-minute segment | `tune_from_feedback.py` |

Per-frame labels are the finer signal but cost hours of annotator time. Teacher
segment ratings cost the teacher a single viewing of their own lesson and three
buttons every five minutes. Both refit the same handful of fields
(`eval/tuning.py:SEARCH_GRID`); use whichever you have, or both.

## End-to-end runbook

1. **Record** the lesson out of band → `backend/recordings/<lesson>.mp4`
   (fixed camera, 1920×1080, the same framing the live camera would use).

2. **Verify pose signs once per camera setup.** `desk_work` detection assumes
   "positive pitch = looking down"; if that is flipped, every fit is garbage.
   First build a cache (step 5), then:
   ```
   python eval/verify_pose_signs.py eval/out/<lesson>.jsonl
   ```
   Must print `PASS` before you trust any tuning output.

3. **Research labels** (optional but recommended):
   ```
   python eval/prepare_annotations.py recordings/<lesson>.mp4 --out annotations/<lesson>
   ```
   Two annotators label independently; check they agree:
   ```
   python eval/agreement.py annotations/<lesson>/a.csv annotations/<lesson>/b.csv
   ```
   A low kappa caps how far any accuracy number can be trusted.

4. **Teacher ratings.** Send the teacher the `.mp4` and the review page link
   (`/review?session=<lesson>`). They watch it back, rate each 5-minute segment
   `focused` / `mixed` / `off` with a sure/unsure flag, then either download
   `segment_ratings.csv` and send it to you, or press "Submit to team" — which
   writes `eval/ratings/<lesson>.csv` server-side in the same format.

5. **Extract** the expensive per-frame measurements once (decode, detect, track,
   pose, gaze). Requires `annotations.csv`; if you only have teacher ratings,
   pass a minimal CSV with a few labelled rows so the harness has something to
   match and `verify_pose_signs.py` has samples.
   ```
   python eval/run_eval.py recordings/<lesson>.mp4 annotations/<lesson>/<lesson>.csv \
       --out eval/out/<lesson>.json
   ```
   This writes the replay cache `eval/out/<lesson>.jsonl`.

6. **Fit.**
   ```
   # against per-frame labels
   python eval/sweep.py eval/out/<lesson>.jsonl annotations/<lesson>/<lesson>.csv \
       --out eval/configs/tuned.yaml --report eval/out/<lesson>.sweep.json

   # against teacher segment ratings
   python eval/tune_from_feedback.py eval/out/<lesson>.jsonl eval/ratings/<lesson>.csv \
       --out eval/configs/tuned.yaml --report eval/out/<lesson>.feedback.json
   ```
   Each prints a baseline-vs-tuned comparison and writes only the fields it
   moved. If nothing beats the baseline it says so and writes the baseline
   values — that is a valid result, not a failure.

7. **Review the report**, commit `eval/configs/tuned.yaml`, and set
   `ATTENTION_CONFIG_PATH=eval/configs/tuned.yaml` in `backend/.env`. The server
   loads it over the `app/config.py` defaults at startup
   (`AttentionConfig.from_settings` → `from_yaml`); no code change.

## Files

| File | Purpose |
|---|---|
| `run_eval.py` | extract (decode→cache) + score (replay cache → accuracy) |
| `prepare_annotations.py` | sample frames, pre-fill detected boxes, write an annotation job |
| `agreement.py` | Cohen's kappa between two annotators |
| `verify_pose_signs.py` | pre-flight check that the head-pose sign convention holds |
| `sweep.py` | fit thresholds to per-frame ground truth (best binary F1) |
| `tune_from_feedback.py` | fit thresholds to teacher segment ratings (smallest band distance) |
| `tuning.py` | shared search grid, rating bands, CSV loader |
| `metrics.py` / `annotations.py` | scoring and label loading |
| `configs/baseline.yaml` | the current assumptions, written out explicitly |
| `configs/tuned.yaml` | fitted overrides — committed, loaded via `ATTENTION_CONFIG_PATH` |
| `out/` | extraction caches and reports (gitignored) |
| `ratings/` | teacher submissions from `/api/eval/ratings` (gitignored) |

## `segment_ratings.csv`

```
session_id,segment_start_s,segment_end_s,rating,confidence,note
5b-0917,0,300,focused,sure,
5b-0917,300,600,mixed,unsure,group work started
```

`rating` ∈ `focused | mixed | off`, mapped to a target class on-task band in
`eval/tuning.py:RATING_BANDS` (`focused` ≥ 0.75, `mixed` 0.40–0.75, `off` < 0.40
— edges tunable there, not in the CSV). `confidence` = `unsure` halves that
row's weight in the fit. `note` is for humans; the fitter ignores it.
