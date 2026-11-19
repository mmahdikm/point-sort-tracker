# Point SORT Tracker

A compact adaptation of SORT for tracking small objects represented by 2D points.

Original development snapshot: **2020-04-06** (last recorded development commit).  
Clean public release: **2026-11-19**.

This repository starts with one public commit containing a cleaned snapshot of earlier work. The development date records the available source history; it is not a new claim of a formal project completion date.

## What changed from SORT

This adaptation uses a four-state constant-velocity Kalman model for position and velocity, two-dimensional observations, Euclidean-distance association, and a distance gate in place of bounding-box overlap. Track age and hit counters retain the SORT lifecycle.

The public snapshot contains the standalone tracking component. Detector integrations, recordings, and experiment-specific data are excluded.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python demo.py
```

The synthetic demo checks identity continuity, a missed observation, and track expiry.

```python
import numpy as np
from tracker import Sort

tracker = Sort(max_age=5, min_hits=1)
tracks = tracker.update(np.array([[100., 50., 0.95]]))
# Each result row is [x, y, track_id].
# Call update once every frame, using np.empty((0, 3)) when nothing is detected.
```

The score column is accepted but association uses position only. The distance threshold is in input-coordinate units. This tracker does not perform appearance-based re-identification or long-occlusion recovery.

## License and attribution

GPL-3.0-or-later. Original SORT copyright and license notices are retained in `tracker.py` and `LICENSE`; see `NOTICE` for the adaptation history. Required upstream attribution is retained without identifying the private application context.
