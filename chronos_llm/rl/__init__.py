"""Shared GRPO utilities (policy-update helpers, checkpoint saving, text-based reward components)."""
from .reward import (
    extract_roi,
    magnitude_band,
    magnitude_match,
    roi_iou,
    text_reward,
)

__all__ = ["extract_roi", "roi_iou", "magnitude_band", "magnitude_match", "text_reward"]
