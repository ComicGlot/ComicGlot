"""Homogeneous-row segmentation (comicglot/utils/seam_segmentor.py)."""
import cv2
import numpy as np

from comicglot.utils.seam_segmentor import calculate_color_homogeneity, segment_by_seam_carving


def loop_homogeneity(image, y, width):
    row = image[y].astype(float)
    diffs = [np.sqrt(np.sum((row[x] - row[x + 1]) ** 2)) for x in range(width - 1)]
    return sum(d < 10 for d in diffs) / (width - 1) * (1.0 if max(diffs) < 50 else 0.0)


def test_vectorised_homogeneity_matches_the_original_loop():
    rng = np.random.default_rng(0)
    for _ in range(50):
        w = int(rng.integers(5, 200))
        img = np.full((3, w, 3), 200, np.uint8)
        img[1] = img[1] + rng.integers(0, 15, (w, 3)).astype(np.uint8)
        assert abs(calculate_color_homogeneity(img, 1, w) - loop_homogeneity(img, 1, w)) < 1e-12


def test_narrow_image_can_be_cut_without_resizing(tmp_path):
    # page 1 is wide, page 2 narrower; both have white gutters. Black padding used to
    # make every row of page 2 fail the test, so page 2 could never be cut.
    wide = np.full((300, 200, 3), 255, np.uint8)
    narrow = np.full((300, 120, 3), 255, np.uint8)
    wide[20:120, 20:180] = 0
    narrow[20:120, 20:100] = 0
    narrow[180:280, 20:100] = 0
    cv2.imwrite(str(tmp_path / "01.png"), wide)
    cv2.imwrite(str(tmp_path / "02.png"), narrow)
    res = segment_by_seam_carving(str(tmp_path), min_height=100, resize=False, carving_step=10)
    cuts = [end for _, end in res["segment_boundaries"][:-1]]
    assert any(300 < c < 600 for c in cuts), cuts
