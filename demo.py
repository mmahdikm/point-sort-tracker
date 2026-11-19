"""Track synthetic point detections, including a missed observation."""
import numpy as np
from tracker import Sort

def main():
    tracker = Sort(max_age=3, min_hits=1)
    first_id = None
    for frame in range(12):
        detections = np.empty((0, 3)) if frame == 5 else np.array([[10. + frame, 20. + .5 * frame, .99]])
        result = tracker.update(detections)
        if frame == 5:
            assert result.shape == (0, 3)
        else:
            assert result.shape == (1, 3)
            if first_id is None:
                first_id = result[0, 2]
            assert result[0, 2] == first_id
            print(frame, result[0].round(3).tolist())
    for _ in range(4):
        tracker.update(np.empty((0, 3)))
    assert not tracker.trackers
    print('Stable identity, missed observation, and track expiry: passed')

if __name__ == '__main__':
    main()
