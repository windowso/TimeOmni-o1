"""Understanding-branch dataset: TimeOmni-v2 style jsonl (Release_train_standard), also
compatible with aggregated datasets that carry reasoning supervision.

Each sample: ``input_text`` is the user turn, ``gt_text``/``gt_result`` the answer, and the time
series is read from ``ori_path`` (csv/npy/wav) as a ``(C, T)`` multi-channel chronos context
(single-channel files -> ``(1, T)``). An optional ``think`` field carries reasoning text (handled
the same way as the ``reasoning`` column in forecast_dataset.py): when non-empty it is routed
through the reasoning path of ``build_supervised_ids``/``build_inference_prefix_ids`` so the
model learns ``<think>reasoning</think>answer``; when missing/empty (the case for all original
TimeOmni-v2 jsonl) the original no-reasoning path is used. The decision is made per sample, so
both kinds of samples may be mixed in one jsonl.
"""

import json
import math
import os
import warnings
from collections import defaultdict
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .chat_utils import (
    _TOKLEN_VERSION,
    build_inference_prefix_ids,
    build_supervised_ids,
    cached_token_lengths,
    measure_llm_text_lengths,
)


def _load_ts_2d(path: str) -> np.ndarray:
    """Read a time-series file and return ``(C, T)`` float32 (multi-channel).

    Convention: raw 2D files are read time-major as ``(T, C)`` (consistent with torchaudio's
    (C,T) followed by a transpose) and transposed to ``(C, T)``; 1D files are treated as a single
    channel ``(1, T)``. Non-finite values become NaN; empty / all-NaN input returns a ``(1, 16)``
    placeholder.
    """
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            arr = np.load(path)
        elif ext == ".csv":
            import pandas as pd

            arr = pd.read_csv(path, header=None).apply(
                lambda c: pd.to_numeric(c, errors="coerce")
            ).values
        elif ext in (".wav", ".mp3", ".flac", ".m4a"):
            # torchaudio 2.10's .load() decodes through the TorchCodec backend, so torchcodec must
            # be installed -- and it **must be pinned to 0.10.0**: 0.11+ is compiled against newer
            # torch (missing symbols such as torch_from_blob, crashes on import), 0.9- does not
            # support system FFmpeg 4; only 0.10.0 is ABI-compatible with torch 2.10 + FFmpeg 4.
            # If it is missing / the wrong version the whole block raises ImportError -> falls into
            # the except branch and degrades to an all-zero placeholder, silently corrupting the
            # signal of audio understanding samples (e.g. Powdermill bird calls). Bumping torch
            # will most likely require bumping this pin as well.
            import torchaudio

            wav, _ = torchaudio.load(path)  # (C, T) float32, already normalised to [-1,1]
            arr = wav.transpose(0, 1).numpy()  # (T, C)
        else:
            raise ValueError(f"unsupported ts ext: {ext}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"failed to load ts {path}: {e}; using placeholder")
        return np.zeros((1, 16), dtype=np.float32)

    arr = np.asarray(arr, dtype=np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if arr.ndim == 1:
        arr = arr[None, :]            # (T,) -> (1, T) single channel
    elif arr.ndim == 2:
        arr = arr.T                   # (T, C) -> (C, T)
    else:
        arr = arr.reshape(arr.shape[0], -1)  # fallback: keep the first axis as channels
    if arr.size == 0:
        return np.zeros((1, 16), dtype=np.float32)
    # Drop **all-NaN channels** (e.g. CSV timestamp/text columns coerced to NaN): every patch of
    # such a row is masked, so a whole row of the temporal self-attention is -inf -> softmax NaN
    # -> soft prompt NaN -> loss NaN.
    ch_valid = ~np.isnan(arr).all(axis=1)
    if not ch_valid.any():
        return np.zeros((1, 16), dtype=np.float32)
    if not ch_valid.all():
        warnings.warn(f"ts {path}: dropping {int((~ch_valid).sum())}/{arr.shape[0]} all-NaN channels")
        arr = arr[ch_valid]
    return np.ascontiguousarray(arr)


def _sample_windows(arr: np.ndarray, k: int, window: int, overview: bool):
    """Uniform window sampling over a very long series + optional whole-series overview
    (a generalisation of truncation: spot checks instead of cutting off the tail).

    arr is (C, T). Not triggered when ``T <= k*window`` (or k<=0): returns ``(arr, None)``
    unchanged, bit-identical to the plain path. When triggered:
    - k native-resolution windows with starts ``round(i*(T-W)/(k-1))`` (first and last always
      included; ``T > k*W`` guarantees they do not overlap); detail inside each window is lossless
      (unlike global downsampling, which destroys high frequencies);
    - with ``overview=True`` one extra chunk is appended: the whole series downsampled to W points
      by **per-bucket nanmean** (box filter, NaN-safe), covering the global shape between windows;
    - the pieces are concatenated along time into ``(C, n*W)`` -- an exact multiple of W, so the
      model's existing chunking loop splits it back into n aligned chunks.
    Returns ``chunk_pos (n, 2) int64``, one row per chunk = (window start, window span) in
    **raw points**: detail windows ``(s, W)``, overview ``(0, T)``. **Must be int64**: under
    DeepSpeed bf16 the HF Trainer ``_prepare_input`` casts every floating tensor in the batch to
    bf16 (at start ~15000 the ulp is 64, which wipes out the high-frequency sinusoidal PE
    components) while integer tensors pass through untouched; the floating-point conversion
    (pos0=start/patch, dpos=span/window) is done on the model side in ``_patch_positions``.
    """
    C, T = arr.shape
    if k <= 0 or T <= k * window:
        return arr, None
    starts = [round(i * (T - window) / (k - 1)) for i in range(k)]
    parts = [arr[:, s:s + window] for s in starts]
    pos = [(s, window) for s in starts]
    if overview:
        bounds = (np.arange(window, dtype=np.int64) * T) // window   # bucket starts (lengths differ by <=1)
        finite = np.isfinite(arr)
        sums = np.add.reduceat(np.where(finite, arr, 0.0), bounds, axis=1)
        cnts = np.add.reduceat(finite.astype(np.float32), bounds, axis=1)
        ov = (sums / np.maximum(cnts, 1.0)).astype(np.float32)
        ov[cnts == 0] = np.nan
        parts.append(ov)
        pos.append((0, T))
    compact = np.ascontiguousarray(np.concatenate(parts, axis=1).astype(np.float32))
    return compact, np.asarray(pos, dtype=np.int64)


class UnderstandingJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_paths: Sequence[str],
        tokenizer,
        base_dir: str | None = None,
        max_user_tokens: int = 1500,
        max_tokens: int = 4096,
        inference_mode: bool = False,
        sample_chunks: int = 0,
        overview_chunk: bool = True,
        sample_window: int = 8192,
        no_reasoning: bool = False,
        answer_first_explanation: bool = False,
        teacher_forced_reasoning: bool = False,
        force_reasoning: bool = False,
        counterfactual_reasoning: bool = False,
        ts_as_text: bool = False,
        answer_prefix: str | None = None,
    ):
        self.data = []
        if isinstance(jsonl_paths, str):
            jsonl_paths = [jsonl_paths]
        # Data-file fingerprint (path/mtime/size), used as part of the token-length cache key.
        self._fingerprint = tuple(
            (p, os.path.getmtime(p), os.path.getsize(p)) for p in jsonl_paths
        )
        n_empty = 0
        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    sample = json.loads(line)
                    # Training: drop empty-answer samples (labels would be all -100 -> CE=NaN);
                    # inference keeps them (gt is handled by eval).
                    if not inference_mode and not self._answer(sample):
                        n_empty += 1
                        continue
                    self.data.append(sample)
        if n_empty:
            warnings.warn(f"understanding jsonl: filtered {n_empty} empty-answer samples")
        self.tokenizer = tokenizer
        self.base_dir = base_dir
        self.max_user_tokens = max_user_tokens
        self.max_tokens = max_tokens
        self.inference_mode = inference_mode
        if sample_chunks == 1:
            raise ValueError("sample_chunks must be 0 (off) or >=2 (the start formula (T-W)/(K-1) needs K>=2)")
        self.sample_chunks = sample_chunks
        self.overview_chunk = overview_chunk
        self.sample_window = sample_window
        # Ablation control (no-CoT): ignore any ``think`` carried by the sample and always take the
        # no-reasoning path => the supervised span is the answer only, and the inference prefix
        # contains the full empty `<think>\n\n</think>\n\n` (the model only continues the answer).
        # See _reasoning().
        self.no_reasoning = no_reasoning
        # Answer-first mode: supervised span = `answer\n\nexplanation` (see
        # _build_item). Mutually exclusive with no_reasoning -- one appends the reasoning to the
        # answer, the other does not want reasoning at all.
        self.answer_first_explanation = answer_first_explanation
        if answer_first_explanation and no_reasoning:
            raise ValueError("answer_first_explanation and no_reasoning are mutually exclusive")
        # **Diagnostic switch, only effective in inference_mode**: feed the sample's own
        # ground-truth reasoning into the prefix so the model only continues the answer.
        # Uses `_reasoning` rather than `_eff_reasoning`: the ground truth does not change with the
        # no_reasoning switch, so this switch also applies to a no-CoT checkpoint. Keep its
        # outputs in a separate directory from the standard evaluation.
        self.teacher_forced_reasoning = teacher_forced_reasoning
        if teacher_forced_reasoning and not inference_mode:
            raise ValueError("teacher_forced_reasoning only makes sense in inference_mode")
        # **Counterfactual probe** (diagnostic): feed not this sample's ground-truth reasoning but
        # the reasoning of **another sample** of the same task, to check whether the model follows
        # the reasoning text or judges from the series itself.
        # When enabled together with teacher_forced_reasoning this switch wins (both stuff
        # reasoning into the prefix; they only differ in whose).
        self.counterfactual_reasoning = counterfactual_reasoning
        if counterfactual_reasoning and not inference_mode:
            raise ValueError("counterfactual_reasoning only makes sense in inference_mode")
        if counterfactual_reasoning:
            # Cyclic shifted pairing within a task: sample i takes the reasoning of sample i+1 of
            # the same task (the last one wraps around to the first). Grouping by task is
            # essential -- reasoning borrowed across tasks would answer a different question, and the
            # probe would only test whether the mismatch is detected.
            by_task = defaultdict(list)
            for i, s in enumerate(self.data):
                if self._reasoning(s):
                    by_task[s.get("task")].append(i)
            self._cf_src = {}
            for idxs in by_task.values():
                for k, i in enumerate(idxs):
                    self._cf_src[i] = idxs[(k + 1) % len(idxs)]
            if any(v == k for k, v in self._cf_src.items()):
                warnings.warn("counterfactual pairing contains self-references (some task has only 1 sample with reasoning); those samples are equivalent to TF")
        # force_reasoning: by default the inference prefix stops at the open `<think>` or contains
        # the full empty think depending on whether *this test sample* has a think. For a test set
        # whose samples carry no think (e.g. ST-Bench, whose official test gives only question and
        # answer) this switch makes the prefix always stop at the open `<think>`, so the model
        # writes its own reasoning, as in reasoning-supervised training.
        # Mutually exclusive with no_reasoning: that means "this run never learns reasoning", so
        # forcing it to write reasoning would be contradictory.
        self.force_reasoning = force_reasoning
        if force_reasoning and no_reasoning:
            raise ValueError("force_reasoning and no_reasoning are mutually exclusive (one forces reasoning, the other never writes it)")
        if force_reasoning and not inference_mode:
            raise ValueError("force_reasoning only makes sense in inference_mode")
        # Forced answer-prefix decoding (**an inference-time intervention, a diagnostic protocol**):
        # the inference prefix continues past the closed think with the fixed opening of the answer
        # (e.g. "Answer:"), so the model can only fill in the answer and cannot write a rationale.
        # See the answer_prefix note in chat_utils.build_inference_prefix_ids. Mutually exclusive
        # with force_reasoning (whose prefix stops at the open <think>).
        self.answer_prefix = (answer_prefix or None)
        if self.answer_prefix and not inference_mode:
            raise ValueError("answer_prefix only makes sense in inference_mode")
        if self.answer_prefix and force_reasoning:
            raise ValueError("answer_prefix and force_reasoning are mutually exclusive (with the prefix stopping at the open <think> the answer opening cannot enter it)")
        # LLM-only baseline: the series is rendered **at full precision as text** into the user
        # text (no downsampling, no digit reduction; anything beyond the token budget is
        # middle-truncated by _truncate_to_tokens keeping both ends, with exactly the same budget
        # as the composite model). history becomes a (1,16) zero placeholder (not consumed by the
        # model in llm_only mode; collator/batch structure unchanged), and sample_chunks window
        # sampling naturally has no effect (it is for the chronos encoder).
        self.ts_as_text = ts_as_text
        self.branch = "understanding"
        self._skip: set[int] = set()  # samples unusable after rendering (too long etc.), discovered lazily in __getitem__

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _ts_len_ch(sample) -> tuple[int, int]:
        """Read (history length, channel count) from fields embedded in the jsonl, without IO;
        missing / non-numeric values yield (0, 1).

        Some jsonl files carry a placeholder string in ``seg_length`` (e.g. ``"DEBUG_SEG_LENGTH"``,
        truthy but not numeric) so a direct ``int()`` would raise ValueError -- convert each field
        defensively and fall through to the next candidate (seg -> ori -> 0) on failure.
        """
        def _num(v) -> int:
            try:
                return int(v)
            except (TypeError, ValueError):
                return 0

        its = sample.get("input_ts") or {}
        if not isinstance(its, dict):
            return 0, 1
        seg = its.get("segment") or {}
        ori = its.get("original") or {}
        length = _num(seg.get("seg_length")) or _num(ori.get("ori_length"))
        ch = _num(its.get("channel")) or 1
        return length, ch

    def _eff_patches(self, length: int, max_context: int, patch: int = 16) -> int:
        """Effective number of time patches after window sampling (exactly the same accounting as
        _sample_windows/_build_item).

        Triggered (K>0 and L > K*W): compact length = (K+overview)*W => the patch count is
        independent of the raw length; not triggered: the plain accounting ceil(min(L, cap)/16).
        The soft-token estimate used by dynamic batching (qformer.num_tokens_for_length(P))
        automatically follows this accounting.
        """
        if self.sample_chunks and length > self.sample_chunks * self.sample_window:
            n = self.sample_chunks + (1 if self.overview_chunk else 0)
            return n * math.ceil(self.sample_window / patch)
        return math.ceil(min(length, max_context) / patch)

    def cost_keys(self, max_context: int = 240_000) -> list:
        """Approximate per-sample compute cost for the batch sampler's length bucketing (no IO /
        tokenisation).

        ~ LLM token count (text chars / 4 as a proxy) + chronos patch count (channels x effective
        patches after window sampling, ``_eff_patches``; the history length is taken from the
        embedded ``ori_length``/``seg_length``, 0 when missing). Bucketing only needs a monotone
        approximation.
        """
        keys = []
        for s in self.data:
            text = len(s.get("input_text", [""])[0]) // 4
            length, ch = self._ts_len_ch(s)
            keys.append(text + ch * self._eff_patches(length, max_context))
        return keys

    def history_patches(self, max_context: int = 240_000, patch: int = 16) -> list:
        """Per-sample (time patches P, channels C), used by dynamic batching to compute soft-token
        counts and the chronos secondary budget."""
        return [
            (self._eff_patches(length, max_context, patch), ch)
            for length, ch in (self._ts_len_ch(s) for s in self.data)
        ]

    def token_lengths(self, cache_dir=None) -> list:
        """Per-sample "number of text tokens fed to the LLM after rendering" (batch tokenisation,
        accurate to within a few tokens), cached on disk (key = data fingerprint + truncation
        parameters + tokenizer + version)."""
        # The key must include inference_mode: training filters empty-answer rows in __init__
        # while inference keeps them, so the row count and order of self.data differ -- reusing
        # the same key would misalign the length array with the samples.
        # The key must include no_reasoning: a CoT and a no-CoT run read the same jsonl files (same
        # fingerprint) but their supervised spans differ by the whole reasoning => hundreds of
        # tokens; sharing the cache would make dynamic batching pack with the wrong cost (badly
        # overestimated on the no-CoT side).
        key = ("understanding", self._fingerprint, self.max_user_tokens, self.max_tokens,
               self.inference_mode, self.no_reasoning, self.answer_first_explanation,
               self.ts_as_text,
               getattr(self.tokenizer, "name_or_path", ""), _TOKLEN_VERSION)
        # If any sample has a think, pass reasoning_texts for the whole batch (empty string for
        # samples without) -- consistent with the scaffold-constant accounting of
        # measure_llm_text_lengths (reasoning_texts=None and all-empty strings yield different
        # scaffolds). **In answer-first mode the lengths must follow the real rendering**: the
        # explanation goes into the **answer** span and the think span is empty. Measuring via the
        # think path would give a similar character count, but the scaffold tokens differ and the
        # dynamic-batching cost estimate would not match the actual rendering (guarded by a unit
        # test).
        if self.answer_first_explanation:
            answers = [self._join_answer_explanation(s) for s in self.data]
            reasoning_texts = None
        else:
            answers = [self._answer(s) for s in self.data]
            has_reasoning = any(self._eff_reasoning(s) is not None for s in self.data)
            reasoning_texts = ([self._eff_reasoning(s) or "" for s in self.data]
                               if has_reasoning else None)

        def _build():
            if not self.ts_as_text:
                return measure_llm_text_lengths(
                    self.tokenizer,
                    user_texts=[s["input_text"][0] for s in self.data],
                    answer_texts=answers,
                    reasoning_texts=reasoning_texts,
                    max_user_tokens=self.max_user_tokens, max_tokens=self.max_tokens,
                )
            # ts_as_text: user text = full-precision series block + question. Samples whose point
            # count already far exceeds the user budget **necessarily hit the cap** max_user_tokens
            # (every value is >=1 token; the threshold is 4x the budget, leaving another 4x safety
            # margin so that even unusual BPE merges of repeated digits cannot underestimate), so
            # record the cap directly and skip the file IO; short-series samples are really
            # rendered (file read) + tokenised for an exact value.
            cap_thresh = 4 * self.max_user_tokens
            user_texts, overrides = [], []
            for s in self.data:
                length, ch = self._ts_len_ch(s)
                if length * ch >= cap_thresh:
                    user_texts.append("")
                    overrides.append(self.max_user_tokens + 8)
                else:
                    user_texts.append(self._user_text_with_ts(s))
                    overrides.append(None)
            return measure_llm_text_lengths(
                self.tokenizer, user_texts=user_texts, answer_texts=answers,
                reasoning_texts=reasoning_texts,
                max_user_tokens=self.max_user_tokens, max_tokens=self.max_tokens,
                user_len_override=overrides,
            )

        return cached_token_lengths(cache_dir, key, _build)

    def _join_answer_explanation(self, sample) -> str:
        """Answer-first supervised content = `answer\\n\\nexplanation` (falls back to the bare
        answer when the explanation is missing).

        `_build_item` and `token_lengths` **share this single joiner** -- writing it twice would
        let the dynamic-batching cost estimate and the actual rendering silently diverge.
        """
        answer = self._answer(sample)
        reasoning = self._eff_reasoning(sample)
        return f"{answer}\n\n{reasoning}" if reasoning else answer

    @staticmethod
    def _answer(sample) -> str:
        gt = sample.get("gt_text") or [""]
        ans = gt[0] if gt and gt[0] else ""
        if ans:
            return str(ans)
        gr = sample.get("gt_result")
        return json.dumps(gr, ensure_ascii=False) if gr else ""

    @staticmethod
    def _reasoning(sample) -> str | None:
        """The sample's own reasoning supervision text; None when missing/empty (original
        no-reasoning path)."""
        think = sample.get("think")
        return str(think).strip() if think and str(think).strip() else None

    def _eff_reasoning(self, sample) -> str | None:
        """Reasoning supervision text actually in effect for this dataset = ``_reasoning``
        filtered through the ``no_reasoning`` switch.

        **Training and inference must share this single entry point**: the training side decides
        whether the supervised span contains `<think>reasoning</think>`, the inference side decides
        whether the prefix stops at the open `<think>` or contains the full empty think -- the two
        must agree, otherwise a no-CoT model would be fed a "continue the reasoning" prefix it
        never learned and generation derails immediately.
        """
        return None if self.no_reasoning else self._reasoning(sample)

    def _prefix_reasoning_text(self, idx, sample):
        """Which reasoning text to put into the inference prefix (diagnostic switches only; returns
        None = original behaviour when neither is on).

        `counterfactual_reasoning` takes precedence over `teacher_forced_reasoning` -- both stuff
        reasoning into the prefix and only differ in whose; when both are on, the counterfactual
        one is the stronger question.
        """
        if self.counterfactual_reasoning:
            src = self._cf_src.get(idx)
            return self._reasoning(self.data[src]) if src is not None else None
        if self.teacher_forced_reasoning:
            return self._reasoning(sample)
        return None

    def _user_text_with_ts(self, sample) -> str:
        """ts_as_text user text = full-precision series text block + original question (series
        first, mirroring where the composite model inserts the soft prompt).

        The pre-cut (a pure rendering/tokenisation cost optimisation) adapts to the channel count:
        each channel keeps ``max(64, ceil(4*max_user_tokens / C))`` points at head and tail =>
        the total rendered amount is <= ~8*max_user_tokens points independent of C (a 12-lead ECG
        with 5000 points/channel no longer renders 60k values). Each rendered head/tail side is
        >= half of 4x the budget >= the head/tail actually kept by token-level middle truncation
        (>=1 token per value, typically 3-4), so what the LLM finally consumes is identical to
        rendering everything and truncating; the middle is replaced by an ellipsis marker stating
        how many points were omitted.
        """
        import math as _math

        from .ts_text import render_series_text

        path = self._ts_path(sample)
        arr = _load_ts_2d(path) if path else np.zeros((1, 16), dtype=np.float32)
        keep = max(64, _math.ceil(4 * self.max_user_tokens / max(1, arr.shape[0])))
        ts_block = render_series_text(arr, max_points_per_side=keep)
        return f"{ts_block}\n\n{sample['input_text'][0]}"

    def _ts_path(self, sample) -> str | None:
        its = sample.get("input_ts")
        # Defensive handling aligned with _ts_len_ch: treat as "no series" when its is not a dict,
        # or when the segment/original key exists but is null (.get(k, {}) only covers a missing
        # key; null is returned as is and the following .get raises AttributeError).
        if not isinstance(its, dict):
            return None
        if its.get("already_segment"):
            path = (its.get("segment") or {}).get("seg_path")
        else:
            path = (its.get("original") or {}).get("ori_path")
        if not path:
            return None
        if self.base_dir and not os.path.isabs(path):
            path = os.path.join(self.base_dir, path)
        return path

    def __getitem__(self, idx):
        # Training: samples that are too long after rendering / have no supervised content
        # (build_supervised_ids returns None) are skipped lazily and replaced by the next usable
        # sample (batch shape unchanged; the _skip cache avoids re-rendering).
        n = len(self.data)
        for off in range(n):
            j = (idx + off) % n
            if j in self._skip:
                continue
            item = self._build_item(j)
            if item is not None:
                return item
            self._skip.add(j)
            warnings.warn(f"understanding sample {j} is too long after rendering / has no supervised content; skipped (replaced by the next sample)")
        raise RuntimeError("no usable sample in the understanding dataset (all too long or without supervised content)")

    def _build_item(self, idx):
        sample = self.data[idx]
        user_text = self._user_text_with_ts(sample) if self.ts_as_text else sample["input_text"][0]
        answer = self._answer(sample)
        reasoning = self._eff_reasoning(sample)  # None (no think / no_reasoning) or non-empty reasoning text

        # **answer-first**: the explanation is appended **after** the answer (newline-separated,
        # not wrapped in `<think>`) and the think span stays empty.
        # `train_answer` and `answer` **must stay separate**: meta["ground_truth"] uses the raw
        # answer, otherwise the ground truth compared by eval would carry the whole explanation
        # and exact matching would always fail.
        train_answer = answer
        if self.answer_first_explanation:
            train_answer = self._join_answer_explanation(sample)   # same joiner as token_lengths
            reasoning = None      # no longer on the think path => prefix identical to no-CoT token by token

        if self.inference_mode:
            # Samples with a non-empty think get a prefix stopping at the open <think> (the model
            # continues reasoning+</think>+answer); samples without think use the original
            # understanding-task prefix (full <think>\n\n</think>\n\n, answer only).
            # With teacher_forced_reasoning the **ground-truth reasoning** is put into the prefix
            # and the model only continues the answer (diagnostic, see below).
            use_reasoning = self.force_reasoning or reasoning is not None
            reasoning_text = self._prefix_reasoning_text(idx, sample)
            if self.answer_prefix and use_reasoning and reasoning_text is None:
                # Do not fail silently: the sample has a think and no_reasoning is off => the prefix
                # stops at the open <think>, the model still writes reasoning first and the answer
                # opening never enters the prefix. Evaluating a test set with think requires
                # no_reasoning to be switched on as well.
                raise ValueError(
                    f"answer_prefix has no effect on sample {sample.get('id', idx)}: it has a think, so the prefix "
                    "stops at the open <think> (the model still writes reasoning first). Switch on no_reasoning "
                    "(or teacher_forced_reasoning) as well")
            input_ids = build_inference_prefix_ids(
                self.tokenizer, user_text=user_text,
                reasoning=use_reasoning,
                max_user_tokens=self.max_user_tokens, max_tokens=self.max_tokens,
                reasoning_text=reasoning_text,
                answer_prefix=self.answer_prefix,
            )
            labels = [-100] * len(input_ids)
        else:
            rendered = build_supervised_ids(
                self.tokenizer,
                user_text=user_text,
                assistant_content=train_answer,
                reasoning_content=reasoning,
                max_user_tokens=self.max_user_tokens,
                max_tokens=self.max_tokens,
            )
            if rendered is None:
                return None
            input_ids, labels = rendered

        if self.ts_as_text:
            # The series is already rendered as text into user_text; history keeps only a (1,16)
            # zero placeholder to preserve the batch structure (not consumed by the llm_only model).
            history = np.zeros((1, 16), dtype=np.float32)
            chunk_pos = None
        else:
            path = self._ts_path(sample)
            history = _load_ts_2d(path) if path else np.zeros((1, 16), dtype=np.float32)
            chunk_pos = None
            if self.sample_chunks:
                history, chunk_pos = _sample_windows(
                    history, self.sample_chunks, self.sample_window, self.overview_chunk)
        item = {
            "branch": "understanding",
            "history": torch.from_numpy(history),
            "input_ids": input_ids,
            "labels": labels,
        }
        if chunk_pos is not None:
            item["chunk_pos"] = torch.from_numpy(chunk_pos)
        if self.inference_mode:
            # Pass through the fields eval needs (aligned with the TimeOmni infer output schema).
            item["meta"] = {
                "id": sample.get("id", str(idx)),
                "uid": sample.get("uid", ""),
                "dataset_name": sample.get("dataset_name", ""),
                "task": sample.get("task", ""),
                "scene": sample.get("scene", ""),
                # Under ts_as_text do not write back the rendered text with the full-precision
                # series block (many rows x tens of thousands of chars would blow up the output
                # jsonl); keep the original question -- the text the model actually consumed is
                # reproducible from the dataset parameters.
                "input_text": sample["input_text"][0] if self.ts_as_text else user_text,
                "ground_truth": answer,
                # or {}: when the jsonl row has "gt_result": null, .get returns None and the eval
                # side, which consumes it as a dict, would crash
                "gt_result": sample.get("gt_result", {}) or {},
                # For the LLM judge to compare against the generated <think> span. **Uses
                # _reasoning rather than _eff_reasoning**: the ground truth does not change with
                # the training configuration -- the no-CoT control group still gets the sample's
                # own think written back, so the gap between "the reasoning that should be there"
                # and the answer it actually produced is visible (it generates no think span, so
                # the comparison naturally scores 0, which is semantically correct).
                "gt_reasoning": self._reasoning(sample) or "",
            }
        return item
