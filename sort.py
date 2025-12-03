import numpy as np

class SimpleTracker:
    def __init__(self, max_distance=50):
        self.next_id = 0
        self.objects = {}
        self.max_distance = max_distance

    def _distance(self, box1, box2):
        x1, y1, x2, y2 = box1
        x3, y3, x4, y4 = box2
        cx1, cy1 = (x1 + x2) / 2, (y1 + y2) / 2
        cx2, cy2 = (x3 + x4) / 2, (y3 + y4) / 2
        return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)

    def update(self, detections):
        updated = []
        used = set()

        for obj_id, old_box in list(self.objects.items()):
            min_dist = 99999
            best = None
            best_idx = -1

            for i, new in enumerate(detections):
                if i in used:
                    continue
                d = self._distance(old_box, new)
                if d < min_dist:
                    min_dist = d
                    best = new
                    best_idx = i

            if min_dist < self.max_distance:
                self.objects[obj_id] = best
                updated.append((obj_id, best))
                used.add(best_idx)

        for i, new in enumerate(detections):
            if i not in used:
                self.objects[self.next_id] = new
                updated.append((self.next_id, new))
                self.next_id += 1

        return updated
