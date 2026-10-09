"""**Textual** rendering of time series for the LLM-only baseline (the numeric series is written out as text and fed
to a plain LLM).

Protocol: **full precision, no downsampling, no reduction of decimal places** -- values use the shortest
round-trip float32 representation (`np.format_float_positional(unique=True)`, parsing back to float32 is
bit-identical), so that the real mechanism "long series blow up the LLM input" is exposed faithfully to the
plain-LLM baseline; whatever exceeds the token budget is handled by `chat_utils._truncate_to_tokens` (cut the
middle, keep both ends), exactly the same text budget as our model.

Two consumers:
- Understanding branch (`UnderstandingJsonlDataset(ts_as_text=True)`): multi-channel rendering channel by channel,
  no timestamps; to save rendering/tokenize cost, ultra-long series are **pre-cropped by point count, keeping head
  and tail** (`max_points_per_side`; guaranteed to leave >= 2x the token budget after pre-cropping, so the token-level
  truncation semantics are unchanged -- the head/tail content the LLM sees equals rendering everything and then
  truncating), with an ellipsis marker in the middle stating how many points were omitted.
- Forecasting branch (`ForecastParquetDataset(ts_as_text=True)`): TimeReasoner-style hybrid rendering
  (timestamp: value, one per line); histories are short and are not pre-cropped; the supervised
  answer gets a trailing `Predicted values:` JSON array (full-precision future values), and at inference
  `parse_forecast_values` parses the values back out of the generated text.
"""

import re

import numpy as np

# The same 21 quantile levels as Chronos-2 (identical to chronos_llm/eval/baseline_timereasoner.py):
# the LLM-only point forecast is broadcast to all 21 quantiles, following the protocol declared for
# ChatTime/TimeReasoner/UniTS (CRPS degenerates to a weighted MAE).
CHRONOS2_QUANTILE_LEVELS = (
    0.01000977, 0.05004883, 0.10009766, 0.15039062, 0.20019531, 0.25, 0.30078125,
    0.34960938, 0.40039062, 0.44921875, 0.5, 0.55078125, 0.6015625, 0.6484375,
    0.69921875, 0.75, 0.80078125, 0.8515625, 0.8984375, 0.94921875, 0.98828125,
)

# Output-format instruction for the forecasting branch (rendered into the user text; fine-tuning and zero-shot use
# the same feed; zero-shot relies entirely on it to constrain the format).
FORECAST_OUTPUT_INSTRUCTION = (
    "After your reasoning and conclusion, output exactly {n} predicted numeric values "
    "for the future horizon as a JSON array on the final line, in the form\n"
    "Predicted values: [v1, v2, ...]"
)


def format_value(v) -> str:
    """Shortest round-trip float32 string (full precision: parsing back to float32 is bit-identical; NaN -> 'nan')."""
    v = np.float32(v)
    if not np.isfinite(v):
        return "nan"
    # unique=True: the shortest decimal representation that uniquely recovers this float32; trim='-' drops trailing
    # zeros and a dangling decimal point.
    return np.format_float_positional(v, unique=True, trim="-")


def _render_1d(vals: np.ndarray, max_points_per_side: int) -> str:
    """Render one channel as comma-separated values; ultra-long series are pre-cropped by point count keeping head and
    tail (the middle is replaced by an ellipsis marker).

    Pre-cropping is only a **rendering-cost** optimisation: the downstream token-level middle truncation
    (`_truncate_to_tokens`) keeps only max_user_tokens/2 tokens at each end, and every value costs at least 1 token,
    so as long as ``max_points_per_side >= max_user_tokens`` the pre-cropped head/tail strictly covers what survives the
    token truncation, and the text the LLM actually sees equals rendering everything and then truncating.
    """
    T = vals.shape[0]
    if max_points_per_side and T > 2 * max_points_per_side:
        head = ", ".join(format_value(v) for v in vals[:max_points_per_side])
        tail = ", ".join(format_value(v) for v in vals[-max_points_per_side:])
        omitted = T - 2 * max_points_per_side
        return f"{head}, ... ({omitted} values omitted) ..., {tail}"
    return ", ".join(format_value(v) for v in vals)


def render_series_text(arr, max_points_per_side: int = 0) -> str:
    """(C, T) or (T,) series -> multi-channel text block (understanding branch, no timestamps).

    A single channel gets no Channel prefix (so it does not clash with the wording of single-channel questions);
    multi-channel input is rendered one line per channel.
    ``max_points_per_side`` is the number of head/tail points kept **per channel** (0 = no pre-cropping).
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    C, T = arr.shape
    header = f"Raw time series ({C} channel{'s' if C > 1 else ''}, {T} points per channel):"
    if C == 1:
        return f"{header}\n{_render_1d(arr[0], max_points_per_side)}"
    lines = [f"Channel {c + 1}: {_render_1d(arr[c], max_points_per_side)}" for c in range(C)]
    return header + "\n" + "\n".join(lines)


def render_forecast_history(values, timestamps=None, freq: str = "") -> str:
    """History block for the forecasting branch: TimeReasoner hybrid style, ``timestamp: value`` per line (plain value
    lines when there are no timestamps).

    The history is rendered in full (the forecasting histories are short, so full-precision text is feasible -- the
    same treatment as the text-LLM baselines; no downsampling or truncation).
    """
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    head = f"Historical time series (frequency={freq}):" if freq else "Historical time series:"
    if timestamps is not None and len(timestamps) == len(vals):
        lines = [f"{t}: {format_value(v)}" for t, v in zip(timestamps, vals)]
    else:
        lines = [format_value(v) for v in vals]
    return head + "\n" + "\n".join(lines)


def forecast_values_block(values) -> str:
    """Future values -> the trailing ``Predicted values: [...]`` of the supervised answer (full precision)."""
    vals = np.asarray(values, dtype=np.float32).reshape(-1)
    return "Predicted values: [" + ", ".join(format_value(v) for v in vals) + "]"


_NUM_RE = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_forecast_values(text: str, horizon: int):
    """Parse the predicted value array out of generated text: take the **last** ``[...]`` array containing >= 1 number.

    Same protocol as baseline_timereasoner: if the array has >= horizon entries, keep the first horizon (rescues
    "a few extra values"); if it is too short, **do not pad with invented values** -- treat it as a parse failure and
    return None, and the caller keeps NaN so the failure is counted truthfully.
    """
    if not text:
        return None
    best = None
    for m in re.finditer(r"\[([^\[\]]*)\]", text, flags=re.S):
        nums = _NUM_RE.findall(m.group(1))
        if nums:
            best = nums
    if best is None or len(best) < horizon:
        return None
    try:
        return np.asarray([np.float32(x) for x in best[:horizon]], dtype=np.float32)
    except (ValueError, OverflowError):
        return None
