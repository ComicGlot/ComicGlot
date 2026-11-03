"""Class-wise NMS used before scoring NMS-free detectors."""
import numpy as np

from eval.detect_eval import apply_nms, nms_classwise


def test_duplicates_removed_and_classes_kept_apart():
    boxes = np.array([[0, 0, .5, .5], [0, 0, .49, .5], [0, 0, .5, .5], [.6, .6, .9, .9]])
    conf = np.array([.9, .8, .7, .6])
    cls = np.array([0, 0, 1, 0])
    keep = nms_classwise(boxes, conf, cls, 0.7)
    assert list(keep) == [0, 2, 3]


def test_apply_nms_keeps_format():
    p = {"f": "a.txt", "xyxyn": [[0, 0, .5, .5], [0, 0, .5, .5]], "cls": [3, 3], "conf": [.2, .9]}
    out = apply_nms([p], 0.7)[0]
    assert out["conf"] == [.9] and out["cls"] == [3] and out["f"] == "a.txt"
