"""Sampling convention of the limited mid-eval subset: equidistant strides covering the whole file,
not the head.

The jsonl files produced by the conversion scripts are arranged in sub-task segments (e.g. HiTSR
L2 before L3), so "take the first `limit` rows" would cover only the first sub-task. Pure
function, no model dependency.
"""
from chronos_llm.eval.infer_understanding import _limit_indices


def test_no_limit_returns_all():
    assert _limit_indices(7, 0) == list(range(7))
    assert _limit_indices(7, -1) == list(range(7))
    assert _limit_indices(7, 7) == list(range(7))
    assert _limit_indices(7, 100) == list(range(7))
    print("limit<=0 / limit>=total returns everything OK")


def test_stride_covers_whole_file():
    idx = _limit_indices(1000, 10)
    assert len(idx) == 10
    assert idx[0] == 0
    assert idx[-1] >= 900, f"last index {idx[-1]} does not reach the tail of the file"
    assert idx == sorted(idx) and len(set(idx)) == 10, "must be ordered and free of duplicates"
    print(f"stride covers the whole file OK: {idx}")


def test_stride_hits_every_subtask_segment():
    """Core scenario: the first 6000 rows of a file are sub-task A and the last 4000 are sub-task B;
    a limit of 300 must pick from both."""
    total, limit, boundary = 10000, 300, 6000
    idx = _limit_indices(total, limit)
    n_a = sum(1 for i in idx if i < boundary)
    n_b = limit - n_a
    assert n_a > 0 and n_b > 0, f"sub-task coverage imbalanced: A={n_a} B={n_b}"
    # Under stride sampling each sub-task's share should be ~ its share of the whole file
    assert abs(n_a / limit - boundary / total) < 0.02, f"share of A {n_a/limit:.3f} deviates from 0.6"
    print(f"both sub-tasks sampled in a segmented file OK: A={n_a} B={n_b}")

    # Control: "take the first `limit` rows" never reaches sub-task B on the same file
    head = list(range(limit))
    assert all(i < boundary for i in head), "construction error"
    print("  (control: taking the first 300 rows gives 0 coverage of sub-task B)")


def test_indices_in_range():
    for total in (1, 2, 5, 37, 1000):
        for limit in (1, 3, 7, 300):
            idx = _limit_indices(total, limit)
            assert all(0 <= i < total for i in idx), (total, limit, idx)
            assert len(idx) == min(limit, total) if limit > 0 else len(idx) == total
    print("indices always within [0,total) and count correct OK")


if __name__ == "__main__":
    test_no_limit_returns_all()
    test_stride_covers_whole_file()
    test_stride_hits_every_subtask_segment()
    test_indices_in_range()
    print("ALL OK")
