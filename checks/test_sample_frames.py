"""Study sampling (eval/sample_frames.py)."""
from collections import Counter

from eval.sample_frames import draw, local_path


def frame(chapter, *types):
    return {"chapter": chapter, "blocks": [{"uid": f"{chapter}/{i}", "class_name": t} for i, t in enumerate(types)]}


def test_rare_types_first_and_limits_kept():
    frames = [frame("a", "speech_bubble")] * 1 + [frame(c, "speech_bubble", "speech_bubble") for c in "abcdef" for _ in range(5)]
    frames = [dict(f) for f in frames]
    frames.append(frame("g", "wiggle_bubble"))
    frames.append(frame("h", *["sfx_sound"] * 9))  # too many blocks: never drawn
    chosen = draw(frames, target=12, per_type=10, max_blocks=6, max_per_chapter=2, seed=0)
    types = Counter(b["class_name"] for i in chosen for b in frames[i]["blocks"])
    chapters = Counter(frames[i]["chapter"] for i in chosen)
    assert types["wiggle_bubble"] == 1                 # the only wiggle frame is taken
    assert types["sfx_sound"] == 0                     # the overloaded frame is not
    assert max(chapters.values()) <= 2                 # chapter cap
    assert sum(types.values()) >= 12                   # target reached


def test_same_seed_same_sample():
    frames = [frame(c, t) for c in "abcd" for t in ("speech_bubble", "sfx_sound", "think_bubble")]
    assert draw(frames, 6, 2, 6, 3, seed=1) == draw(frames, 6, 2, 6, 3, seed=1)


def test_local_path_keeps_from_data_on(tmp_path):
    p = local_path(r"E:\old\place\data\output\run\segments\segment_000.png", tmp_path)
    assert p == tmp_path / "data" / "output" / "run" / "segments" / "segment_000.png"
