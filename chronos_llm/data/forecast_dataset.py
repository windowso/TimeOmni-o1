"""Forecasting-branch dataset: the event-caption parquet corpus.

Each sample: history_values / future_values (univariate), background+event+prompt as the user
text, reasoning rendered inside `<think>`, conclusion as the answer; the ROI given by
roi_start/end (absolute coordinates, minus past_len) is converted into a mask relative to the
future window.
"""

import math
import warnings

import numpy as np
import torch
from torch.utils.data import Dataset

import os

from .chat_utils import (
    _TOKLEN_VERSION,
    build_inference_prefix_ids,
    build_prompt_only_ids,
    build_supervised_ids,
    cached_token_lengths,
    measure_llm_text_lengths,
)


def _np_series(cell) -> np.ndarray:
    """Convert a parquet cell into a float32 ndarray, robust to nested multi-channel lists:
    - 1D ``list<float>`` -> ``(T,)``;
    - 2D ``list<list<float>>`` (pandas reads it back as an object array of arrays) -> stacked into ``(C, T)``.
    """
    a = np.asarray(cell)
    if a.dtype == object:
        a = np.stack([np.asarray(x, dtype=np.float32) for x in a])
    return np.asarray(a, dtype=np.float32)


class ForecastParquetDataset(Dataset):
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        split: str | None = "train",
        max_user_tokens: int = 1500,
        max_tokens: int = 4096,
        max_rows: int | None = None,
        inference_mode: bool = False,
        forecast_prompt_only: bool = False,
        emit_meta: bool | None = None,
        emit_ss_prefix: bool = False,
        ts_as_text: bool = False,
    ):
        import pandas as pd

        if max_rows is not None:
            # Save memory: read only as many row groups as needed (used by smoke tests).
            import pyarrow as pa
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(parquet_path)
            tables, n = [], 0
            for rg in range(pf.num_row_groups):
                t = pf.read_row_group(rg)
                tables.append(t)
                n += t.num_rows
                if n >= max_rows:
                    break
            df = pa.concat_tables(tables).to_pandas()
        else:
            df = pd.read_parquet(parquet_path)
        if split is not None:
            if "split" in df.columns:
                df = df[df["split"] == split]
            else:
                warnings.warn(
                    f"parquet has no split column; split={split!r} silently degrades to all {len(df)} rows -- "
                    "if this is used for evaluation, make sure no training rows are mixed in (metrics would be inflated by leakage)")
        if not inference_mode and "conclusion" in df.columns:
            # Training: drop rows with an empty conclusion (empty supervised answer -> labels may be all -100 -> CE=NaN).
            # notna must come first: null (None/NaN) becomes the truthy text 'None'/'nan' after astype(str)
            # and would slip through the filter, training on a literal 'None' answer / 'nan' reasoning.
            keep = df["conclusion"].notna() & df["conclusion"].astype(str).str.strip().astype(bool)
            if "reasoning" in df.columns:
                keep &= df["reasoning"].notna() & df["reasoning"].astype(str).str.strip().astype(bool)
            if (~keep).any():
                warnings.warn(f"forecast parquet: dropping {int((~keep).sum())} rows with empty conclusion/reasoning")
                df = df[keep]
        if max_rows is not None:
            df = df.head(max_rows)
        self.df = df.reset_index(drop=True)
        # Data-file fingerprint (path/mtime/size + row-selection parameters), used as part of the token-length cache key.
        self._fingerprint = (parquet_path, os.path.getmtime(parquet_path),
                             os.path.getsize(parquet_path), split, max_rows)
        self.tokenizer = tokenizer
        self.max_user_tokens = max_user_tokens
        self.max_tokens = max_tokens
        self.inference_mode = inference_mode
        # In-training scheduled sampling: additionally emit an inference-prefix id sequence next to the
        # training rendering (training mode only).
        self.emit_ss_prefix = emit_ss_prefix and not inference_mode and not forecast_prompt_only
        # Whether to pass through per-sample meta (id/dataset_name, ...). Defaults to inference_mode (the
        # training rendering emits none, zero impact on the training path); the teacher-forced evaluation
        # uses the training rendering (inference_mode=False) but still needs the real id/dataset_name to
        # align metrics -> switch on explicitly with emit_meta=True (otherwise infer falls back to the row
        # number and the ids in the tf npz degrade to ordinals).
        self.emit_meta = inference_mode if emit_meta is None else emit_meta
        # Ablation 3 (prompt_only): build prompt-only ids (labels all -100) from plain_prompt (a condensed
        # prompt without the reasoning instruction); neither training nor evaluation generates reasoning,
        # only the prompt-text hidden states are fed back.
        self.forecast_prompt_only = forecast_prompt_only
        if forecast_prompt_only and "plain_prompt" not in self.df.columns:
            raise ValueError("forecast_prompt_only=True requires the parquet to contain a plain_prompt column"
                             " (generate it with scripts/utils/build_plain_prompt.py --apply)")
        # LLM-only baseline: the **full, full-precision** history is textualised (TimeReasoner hybrid style,
        # one `timestamp: value` line per step + an output-format instruction) and appended to the user text;
        # the supervised answer = conclusion + the full-precision future value array.
        # Note on lengths: the histories of this corpus are short, so the history text needs no truncation --
        # but the caller must enlarge max_user_tokens/max_tokens (train_forecast.sh does so under LLM_ONLY=1),
        # otherwise whole samples are skipped as over-length.
        self.ts_as_text = ts_as_text
        if ts_as_text and forecast_prompt_only:
            raise ValueError("ts_as_text and forecast_prompt_only are mutually exclusive")
        self.branch = "forecast"
        self._skip: set[int] = set()  # samples unusable after rendering (over-length etc.), discovered lazily in __getitem__

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _hist_len_ch(row) -> tuple[int, int]:
        """Return (history length, number of channels) from the in-memory history_values."""
        h = row["history_values"]
        if len(h) and np.ndim(h[0]) > 0:   # multi-channel (C, T)
            return len(h[0]), len(h)
        return len(h), 1                   # single channel (T,)

    def cost_keys(self, max_context: int = 8192) -> list:
        """Approximate per-sample compute cost for the batch sampler's length bucketing (no tokenisation).

        ~ LLM token count (characters of background/event/prompt/reasoning/conclusion divided by 4 as a
        proxy) + chronos patch count (channels x ceil(truncated history / 16)). Bucketing only needs a
        monotone approximation.
        """
        keys = []
        for _, row in self.df.iterrows():
            text = sum(
                len(str(row[c])) for c in ("background", "event", "prompt", "reasoning", "conclusion")
                if c in row and row[c] is not None
            ) // 4
            t, ch = self._hist_len_ch(row)
            keys.append(text + ch * math.ceil(min(t, max_context) / 16))
        return keys

    def history_patches(self, max_context: int = 8192, patch: int = 16) -> list:
        """Per-sample (number of time patches P, number of channels C), used by dynamic batching to
        compute the soft-token count and the chronos secondary budget."""
        out = []
        for _, row in self.df.iterrows():
            t, ch = self._hist_len_ch(row)
            out.append((math.ceil(min(t, max_context) / patch), ch))
        return out

    def token_lengths(self, cache_dir=None) -> list:
        """Per-sample "number of text tokens fed to the LLM after rendering" (batch tokenisation, accurate
        to within a few tokens), cached on disk (key = data-file fingerprint + truncation parameters +
        tokenizer + version)."""
        # The key must include inference_mode / forecast_prompt_only: different modes change row counts / text lengths.
        key = ("forecast", self._fingerprint, self.max_user_tokens, self.max_tokens,
               self.inference_mode, self.forecast_prompt_only, self.ts_as_text,
               getattr(self.tokenizer, "name_or_path", ""), _TOKLEN_VERSION)
        users = [self._user_text(row) for _, row in self.df.iterrows()]
        if self.forecast_prompt_only:
            answers, reasonings = [""] * len(users), None  # prompt-only: no answer/reasoning segment
        else:
            answers = [self._train_answer(row) for _, row in self.df.iterrows()]
            reasonings = [str(row["reasoning"]).strip() for _, row in self.df.iterrows()]
        return cached_token_lengths(cache_dir, key, lambda: measure_llm_text_lengths(
            self.tokenizer, user_texts=users, answer_texts=answers, reasoning_texts=reasonings,
            max_user_tokens=self.max_user_tokens, max_tokens=self.max_tokens,
        ))

    def _train_answer(self, row) -> str:
        """The supervised answer segment for training: the conclusion; with ts_as_text, the full-precision
        future value array is appended (this text *is* the LLM-only baseline's "quantile head" -- reasoning,
        then conclusion, then the numbers, structurally the same supervision as the FULL arm).

        Shared with ``token_lengths`` (the cost estimate for dynamic batching must not diverge from the
        actual rendering; same lesson as understanding._join_answer_explanation).
        """
        conclusion = str(row["conclusion"]).strip()
        if not self.ts_as_text:
            return conclusion
        from .ts_text import forecast_values_block

        fut = _np_series(row["future_values"])
        if fut.ndim == 2:
            fut = fut[0]
        return f"{conclusion}\n\n{forecast_values_block(fut)}"

    @staticmethod
    def _clean_text(v) -> str:
        """Missing parquet values read back as None/NaN(float): NaN is truthy and str() turns it into the
        literal text 'nan' inside the prompt -- normalise everything to the empty string."""
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return ""
        return str(v).strip()

    def _user_text(self, row) -> str:
        if self.ts_as_text:
            # LLM-only: prompt (complete, self-contained task description) + the full, full-precision
            # history (one `timestamp: value` line per step) + the output-format instruction. For a
            # multi-channel history the first channel is used (consistent with the convention of the
            # TimeReasoner/ChatTime text baselines; this corpus is in fact univariate).
            from .ts_text import FORECAST_OUTPUT_INSTRUCTION, render_forecast_history

            hist = _np_series(row["history_values"])
            if hist.ndim == 2:
                hist = hist[0]
            ts = row.get("history_timestamps")
            ts = list(ts) if ts is not None and np.ndim(ts) > 0 else None
            fl = int(row.get("future_len") or len(row["future_values"]))
            return (f"{self._clean_text(row.get('prompt'))}\n\n"
                    f"{render_forecast_history(hist, ts, self._clean_text(row.get('freq')))}\n\n"
                    f"{FORECAST_OUTPUT_INSTRUCTION.format(n=fl)}")
        if self.forecast_prompt_only:
            # Condensed plain prompt (background+event+metadata, reasoning instruction removed); already a
            # complete, fluent paragraph on its own.
            return self._clean_text(row.get("plain_prompt"))
        # The prompt column is itself a complete, self-contained task description (it carries all of the
        # "## Dataset background"/"## Event (established fact)"/"## Series metadata"/"## Task" sections, as
        # written when the corpus was exported) -- there is no need to prepend the
        # background/event columns again (they would appear twice in the final text).
        return self._clean_text(row.get("prompt"))

    def __getitem__(self, idx):
        # Training: samples that are over-length after rendering / have no supervised content
        # (build_supervised_ids returns None) are skipped lazily and the next usable sample is taken
        # instead (batch shape unchanged; the _skip cache avoids re-rendering).
        n = len(self.df)
        for off in range(n):
            j = (idx + off) % n
            if j in self._skip:
                continue
            item = self._build_item(j)
            if item is not None:
                return item
            self._skip.add(j)
            warnings.warn(f"forecast sample {j} is over-length after rendering / has no supervised content; skipping (using the next sample instead)")
        raise RuntimeError("no usable sample in the forecast dataset (all over-length or without supervised content)")

    def _build_item(self, idx):
        row = self.df.iloc[idx]
        # Single channel: history_values/future_values are 1D; multi-channel: history (C1,T), future (n_targets,fl),
        # channel row order convention [targets, known-future cov, past-only cov]; optional column future_covariates (n_fut,fl).
        # Time-series hygiene (the understanding branch falls back to placeholders in _load_ts_2d; in the forecasting
        # branch the row order is bound to target/covariate roles, channels cannot be dropped, so a bad row can only
        # skip the whole sample via the lazy-skip fallback): a parse failure / empty series / all-NaN channel makes
        # every patch of that chronos row masked -> the group-attention row is all -inf -> NaN poisons the whole batch.
        try:
            history = _np_series(row["history_values"])
            future = _np_series(row["future_values"])
        except Exception as e:  # dirty data such as ragged nested lists: np.stack's exception would not be caught by the lazy skip
            warnings.warn(f"forecast sample {idx}: failed to parse the time series ({type(e).__name__}: {e}); skipping")
            return None
        h2 = history if history.ndim == 2 else history[None, :]
        f2 = future if future.ndim == 2 else future[None, :]
        if (history.size == 0 or future.size == 0
                or np.isnan(h2).all(axis=-1).any() or np.isnan(f2).all(axis=-1).any()):
            warnings.warn(f"forecast sample {idx}: history/future is empty or contains an all-NaN channel; skipping")
            return None
        fl = future.shape[-1]
        ss_prefix = None

        if self.forecast_prompt_only:
            # Ablation 3: both training and evaluation produce plain-prompt-only ids (no think/reasoning/conclusion),
            # labels all -100 (no text CE). The model forward feeds back the hidden states of the whole sequence
            # (soft + plain prompt).
            input_ids = build_prompt_only_ids(
                self.tokenizer, user_text=self._user_text(row),
                max_user_tokens=self.max_user_tokens, max_tokens=self.max_tokens,
            )
            labels = [-100] * len(input_ids)
        elif self.inference_mode:
            # Inference: feed only the prefix (stopping at the opening <think>; the model continues with
            # reasoning+</think>+conclusion); labels are an all -100 placeholder (unused by generate);
            # future/roi are still passed through as ground truth for the metrics.
            input_ids = build_inference_prefix_ids(
                self.tokenizer, user_text=self._user_text(row), reasoning=True,
                max_user_tokens=self.max_user_tokens, max_tokens=self.max_tokens,
            )
            labels = [-100] * len(input_ids)
        else:
            rendered = build_supervised_ids(
                self.tokenizer,
                user_text=self._user_text(row),
                assistant_content=self._train_answer(row),
                reasoning_content=str(row["reasoning"]).strip(),
                max_user_tokens=self.max_user_tokens,
                max_tokens=self.max_tokens,
            )
            if rendered is None:
                return None
            input_ids, labels = rendered
            if self.emit_ss_prefix:
                # In-training scheduled sampling: additionally render the inference prefix (stopping at the
                # opening <think>, exactly the same layout as evaluation-time generate) so the model can
                # self-generate reasoning+conclusion on SS batches as the feedback condition.
                ss_prefix = build_inference_prefix_ids(
                    self.tokenizer, user_text=self._user_text(row), reasoning=True,
                    max_user_tokens=self.max_user_tokens, max_tokens=self.max_tokens,
                )

        # ROI absolute coordinates -> coordinates relative to the future window. Null/NaN coordinate columns
        # are treated as "no ROI": the ValueError from int(NaN) would not be caught by the lazy skip and
        # would kill the training/evaluation process outright.
        def _safe_int(v):
            try:
                f = float(v)
            except (TypeError, ValueError):
                return None
            return int(f) if np.isfinite(f) else None

        past_len = _safe_int(row["past_len"])
        rs_abs = _safe_int(row["roi_start_idx"])
        re_abs = _safe_int(row["roi_end_idx"])
        roi = np.zeros(fl, dtype=np.float32)
        if past_len is not None and rs_abs is not None and re_abs is not None:
            rs = max(0, min(rs_abs - past_len, fl))
            re = max(0, min(re_abs - past_len, fl))
            if re > rs:
                roi[rs:re] = 1.0

        item = {
            "branch": "forecast",
            "history": torch.from_numpy(history),
            "future": torch.from_numpy(future),
            "roi": torch.from_numpy(roi),
            "input_ids": input_ids,
            "labels": labels,
        }
        if ss_prefix is not None:
            item["ss_prefix_ids"] = ss_prefix
        if self.emit_meta:
            # Pass through the id and the valid horizon length (the ground truth is judged by non-NaN future
            # values; the raw length is recorded here for alignment).
            # input_text = the user text fed to the model (background+event+prompt); gt_reasoning/gt_conclusion
            # = the real text ground truth, so the inference jsonl lets a human compare "self-generated
            # reasoning/conclusion in gen_text vs the reference".
            item["meta"] = {
                "id": str(row["id"]) if "id" in row else str(idx),
                "dataset_name": str(row.get("dataset_name", "")),
                "future_len": int(future.shape[-1]),
                "input_text": self._user_text(row),
                "gt_reasoning": self._clean_text(row.get("reasoning")),
                "gt_conclusion": self._clean_text(row.get("conclusion")),
            }
        # Known-future covariates (optional column): (n_fut, fl) or 1D.
        fc = row.get("future_covariates") if hasattr(row, "get") else None
        if fc is not None and np.ndim(fc) > 0:   # skip a missing column (None) / scalar NaN
            fc = _np_series(fc)
            if fc.size > 0:
                item["future_covariates"] = torch.from_numpy(fc if fc.ndim == 2 else fc[None, :])
        return item
