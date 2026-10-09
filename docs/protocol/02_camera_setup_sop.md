# SOP 02 — Two-smartphone camera setup (DRAFT)

> DRAFT for team review. Distances and heights are placeholders until piloted (M3 test clips).

## Equipment
Two smartphones on tripods (frontal + sagittal), 60 fps minimum, 120 fps preferred, fixed focus
and exposure, landscape orientation. Record the phone model and fps for every session
(`screening_sessions.camera_setup`).

## Placement
- **Frontal camera:** facing the athlete, optical axis perpendicular to the frontal plane.
- **Sagittal camera:** to the side of the test limb, perpendicular to the sagittal plane.
- Lens height, distance, and floor markings: TODO(human) fix after pilot; keep identical across
  sessions and sites.
- Whole body in frame for the entire task, including the drop box and landing area.

## Synchronisation
One clear clap (audio) or LED flash visible to both cameras at the start of each trial; used by
`pose/sync.py` for cross-correlation.

## Clothing and environment
Fitted shorts, shoes as worn for training (record footwear), even lighting, uncluttered
background, no other people in frame.

## Privacy
Video is copied to the local `data/raw_video_LOCAL_ONLY/` directory of the processing machine
only, named by pseudonymous ID. It is never uploaded to cloud storage, messaging apps, or the
repository, and is deleted from the phones after transfer. Retention period: TODO(human).
