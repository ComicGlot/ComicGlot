"""Reading order and overlap separators (comicglot/utils/spatial_processor.py)."""
import numpy as np

from comicglot.utils.data_types import TextBlock
from comicglot.utils.spatial_processor import apply_separator_mask, create_separator_line, process_text_blocks


def blocks(*boxes):
    return [TextBlock(box=np.array(b, float), class_name="speech_bubble", confidence=0.9, id=i) for i, b in enumerate(boxes)]


def order(bs):
    sorted_blocks, _ = process_text_blocks(bs)
    return [tuple(map(int, b.box)) for b in sorted_blocks]


def test_side_by_side_balloons_read_left_to_right_then_down():
    right, left, below = (200, 15, 300, 95), (10, 10, 110, 90), (50, 300, 150, 380)
    assert order(blocks(right, left, below)) == [left, right, below]


def test_lower_balloon_in_same_band_starts_a_new_line():
    # overlaps vertically with the first box (same group) but its top is more than
    # half the box height lower, so it is read after, even though it sits to the left
    top, lower_left = (200, 0, 300, 100), (0, 60, 100, 160)
    assert order(blocks(top, lower_left)) == [top, lower_left]


def test_crossing_boxes_get_a_two_point_separator():
    line = create_separator_line(np.array([0, 0, 100, 100]), np.array([50, 50, 150, 150]))
    assert line is not None and line.shape == (2, 2)


def test_disjoint_boxes_get_no_separator():
    assert create_separator_line(np.array([0, 0, 10, 10]), np.array([20, 20, 30, 30])) is None


def test_mask_keeps_the_side_with_the_crop_centre():
    img = np.full((100, 100, 3), 255, np.uint8)
    out = apply_separator_mask(img, np.array([[80, 0], [80, 99]]))
    assert out[50, 50].sum() > 0          # centre side kept
    assert out[50, 95].sum() == 0         # far side blacked out
