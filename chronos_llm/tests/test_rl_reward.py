"""Unit tests of the GRPO reward components (pure text parsing + arithmetic, CPU, no torch)."""
from chronos_llm.rl.reward import (
    extract_roi,
    magnitude_band,
    magnitude_match,
    roi_iou,
    text_reward,
)


def test_extract_roi():
    assert extract_roi("The span [508, 546) shows a broad bell") == (508, 546)
    assert extract_roi("The ROI spans rows 510 through 551, covering") == (510, 552)
    assert extract_roi("no roi here") is None
    print("extract_roi OK")


def test_roi_iou():
    assert roi_iou((0, 10), (0, 10)) == 1.0
    assert roi_iou((0, 10), (10, 20)) == 0.0          # no overlap
    assert abs(roi_iou((0, 10), (5, 15)) - 5 / 15) < 1e-9   # intersection 5 / union 15
    assert roi_iou(None, (0, 10)) == 0.0
    print("roi_iou OK")


def test_magnitude_band():
    assert magnitude_band("at a small fraction of the usual highs") == -2
    assert magnitude_band("runs well below the typical highs") == -1
    assert magnitude_band("climbs above the usual level") == 1
    assert magnitude_band("a broad smooth bell rises to a crest") == 0   # no magnitude word
    print("magnitude_band OK")


def test_magnitude_match():
    assert magnitude_match("small fraction", "small fraction") == 1.0     # same band
    assert magnitude_match("well below", "small fraction") == 0.5         # one band apart (-1 vs -2)
    assert magnitude_match("well above", "small fraction") == 0.0         # opposite direction (1 vs -2)
    assert magnitude_match("smooth bell", "smooth bell") == 1.0           # both = 0 typical
    print("magnitude_match OK")


def test_text_reward():
    # Generated ROI slightly off + magnitude direction right -> medium-high score
    total, d = text_reward(
        "The ROI spans rows 508 through 545. Overall, at a small fraction of the usual highs.",
        gt_roi=(508, 546),
        gt_text="[508, 546) suppressed. Overall, a small fraction of the series' usual highs.",
    )
    assert d["roi_iou"] > 0.9 and d["mag_match"] == 1.0, d
    # Magnitude direction opposite -> mag_match=0, lowers the total
    total2, d2 = text_reward(
        "[508, 546). Overall, climbs well above the usual level.",
        gt_roi=(508, 546),
        gt_text="Overall, a small fraction of the usual highs.",
    )
    assert d2["mag_match"] == 0.0 and total2 < total, (d2, total2, total)
    print("text_reward OK")


if __name__ == "__main__":
    test_extract_roi()
    test_roi_iou()
    test_magnitude_band()
    test_magnitude_match()
    test_text_reward()
    print("ALL RL REWARD TESTS PASSED")
