# M3 status — 2026-10-10: pipeline built and tested from keypoints onward; real-video run NOT done

## Acceptance criteria
| Criterion | Result |
|---|---|
| Angle-convention tests | ✅ `test_angle_conventions.py`: knee flexion 0° at full extension and equal to 180 − included angle; hip, ankle, pelvic drop, trunk lean conventions on hand-built geometries |
| Landmark tests | ✅ `test_landmark_indices.py`: BlazePose lower limb is 23–32; code and config agree |
| Filtering tests | ✅ `test_filtering.py`: gap rules; −3 dB at the cut-off for 6, 8, 12 Hz; zero phase lag; deg/s from true (uneven) timestamps; gaps not bridged |
| Event-asymmetry tests | ✅ `test_event_asymmetry.py`: asymmetry only from event/window features; unilateral tasks compare left-limb trials with right-limb trials |
| FPPA test | ✅ `test_fppa_synthetic_skeleton.py`: recovered on both limbs, positive for valgus, unchanged between 40° and 110° of knee flexion |
| Runs end-to-end on a short local test clip | ❌ **Not done.** No test clip and no MediaPipe model file are available. The pose extra (MediaPipe, OpenCV, rtmlib) was not installed, so the two estimator wrappers have never been executed. `TODO(human)`: provide 2–3 consented or self-recorded adult clips and the model file |
| `make test lint` green | ✅ 245 passed, 2 deselected; `mypy --strict` clean on 47 files |

## What is and is not verified
- **Verified on synthetic sequences** (stylised stick figure with exactly known angles and event
  frames, plus pixel noise, dropped frames, missing landmarks, and a clock offset between
  cameras): keypoint storage, gap filling, filtering, angles, events, features, session
  aggregation, asymmetry, quality control and the exclusion log.
- **Not verified at all**: `estimators/mediapipe.py`, `estimators/rtmpose.py`, and
  `scripts/run_pose.py`. They are written to the documented APIs and excluded from the coverage
  figure. The Halpe-26 landmark indices used for RTMPose were written from memory.
- **Synthetic accuracy says nothing about real accuracy.** Recovering angles to within about 1°
  on a clean stick figure shows the arithmetic and conventions are right. Agreement with a
  marker-based system on children is Paper A's question (M4 and the lab study).

## Known behaviour to be aware of
- Contact events are found by a height threshold, so initial contact is detected a few frames
  before true touch-down and take-off a few frames after (within 5 frames at 120 fps on
  synthetic data). Thresholds in `configs/pose/events.yaml` are assumptions awaiting real clips.
- Single-leg-squat start and return are defined as crossing standing knee flexion plus an onset
  threshold, which is later than the first visible movement.
- Angles are 2D projections per camera. Out-of-plane movement biases them; camera placement in
  `docs/protocol/02_camera_setup_sop.md` matters.
- Ankle dorsiflexion has a constant offset from landmark placement; use change from standing.
- Tuck-jump features are not implemented; such trials are logged as unsupported.
- Extracting the audio or brightness trace for two-phone synchronisation from video files is
  not implemented (the alignment itself is).

## Next: M4
Validation module for Paper A (agreement statistics against marker-based mocap).
