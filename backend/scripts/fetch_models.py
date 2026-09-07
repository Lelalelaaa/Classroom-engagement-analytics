#!/usr/bin/env python3
"""
Download the model weights the pipeline needs.

Weights are gitignored rather than vendored, so a fresh clone runs this once:

    python scripts/fetch_models.py

The MediaPipe face_landmarker.task is not listed here — mediapipe_init fetches
it on first use.
"""
import os
import sys
import urllib.request

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

MODELS = [
    (
        "models/face-detection/face_detection_yunet_2023mar.onnx",
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
        "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "YuNet face detector",
    ),
    (
        "models/gaze-estimation/weights/resnet34_gaze.onnx",
        "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet34_gaze.onnx",
        "Eye gaze (resnet34)",
    ),
    (
        "models/gaze-estimation/weights/resnet18_gaze.onnx",
        "https://github.com/yakhyo/gaze-estimation/releases/download/weights/resnet18_gaze.onnx",
        "Eye gaze (resnet18, faster fallback)",
    ),
]

MIN_BYTES = 10_000   # anything smaller is an error page or a git-lfs pointer


def main() -> int:
    failed = []
    for rel_path, url, label in MODELS:
        dest = os.path.join(BACKEND, rel_path)
        if os.path.exists(dest) and os.path.getsize(dest) > MIN_BYTES:
            print(f"[ok]   {label} — already present")
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print(f"[..]   {label} — downloading")
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"[FAIL] {label}: {exc}")
            failed.append(label)
            continue
        size = os.path.getsize(dest)
        if size <= MIN_BYTES:
            print(f"[FAIL] {label}: got {size} bytes — check the URL")
            os.remove(dest)
            failed.append(label)
        else:
            print(f"[ok]   {label} — {size / 1e6:.1f} MB")

    if failed:
        print(f"\n{len(failed)} download(s) failed: {', '.join(failed)}")
        return 1
    print("\nAll models present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
