"""Build LLM inputs: insert the <ts_start><ts_end> time-series placeholder in front of the user
text, and teacher-force the assistant answer using the longest-common-prefix (LCP) trick so that only
the actual answer is supervised (reasoning + conclusion for the forecast branch, the answer text for
the understanding branch), masking the user turn / assistant header / `<think>` scaffolding.

Reasoning is rendered into `<think>...</think>` through the ``reasoning_content`` field natively
supported by the Qwen3.5 chat template; the conclusion follows as content.
"""

import os

from chronos_llm.models.chronos_llm_model import TS_START, TS_END


def wrap_user_with_ts(user_text: str) -> str:
    """Place the time-series placeholder at the very beginning of the user content (the soft prompt
    is injected after <ts_start> in forward).

    Literal <ts_start>/<ts_end> occurrences inside the user text (leftovers from web/CSV cleaning)
    would be tokenised as special tokens and hijack the insertion point (_splice_soft_prompt takes
    the **first** <ts_start>) -- strip them from the body first."""
    user_text = user_text.replace(TS_START, "").replace(TS_END, "")
    return f"{TS_START}{TS_END}\n{user_text}"


def _truncate_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate an over-long text to about ``max_tokens`` tokens by **cutting the middle and keeping
    both ends**.

    The beginning (background/instructions) and the end (question/options) of the user text both
    usually carry essential information; cutting either end can leave the supervised answer without
    grounding, so keep half at each end and drop the middle (a truncation marker is inserted, which
    adds a few tokens).
    """
    if max_tokens is None:
        return text
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    head = max_tokens // 2
    tail = max_tokens - head
    return (
        tokenizer.decode(ids[:head], skip_special_tokens=True)
        + "\n... (truncated) ...\n"
        + tokenizer.decode(ids[-tail:], skip_special_tokens=True)
    )


def _lcp_len(a: list, b: list) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def build_supervised_ids(
    tokenizer,
    user_text: str,
    assistant_content: str,
    reasoning_content: str | None = None,
    max_user_tokens: int = 1500,
    max_tokens: int = 4096,
):
    """Return (input_ids, labels); return ``None`` when the sample is unusable (the caller skips it).

    The answer start is located as the longest common prefix of the "empty answer rendering" and the
    "real answer rendering", so it is independent of the specific chat template / think scaffolding
    and follows template changes automatically.

    Two kinds of samples return None instead of silently producing bad labels:
    - **Over-long**: the whole rendering exceeds max_tokens. Tail truncation would silently cut the
      conclusion/answer (teaching the model to write reasoning that never ends); in the extreme the
      whole supervised span is cut -> labels all -100 -> CE=NaN.
    - **Nothing to supervise**: empty answer (LCP == full length) -> labels all -100 -> CE=NaN; a
      single such sample poisons the whole batch.
    """
    user_text = _truncate_to_tokens(tokenizer, user_text, max_user_tokens)
    user = {"role": "user", "content": wrap_user_with_ts(user_text)}

    full_msg = {"role": "assistant", "content": assistant_content}
    empty_msg = {"role": "assistant", "content": ""}
    if reasoning_content is not None:
        full_msg["reasoning_content"] = reasoning_content
        empty_msg["reasoning_content"] = ""

    full = tokenizer.apply_chat_template([user, full_msg], tokenize=True, return_dict=False)
    if len(full) > max_tokens:
        return None
    empty = tokenizer.apply_chat_template([user, empty_msg], tokenize=True, return_dict=False)
    prefix_len = _lcp_len(full, empty)
    input_ids = list(full)
    labels = [-100] * prefix_len + list(full[prefix_len:])
    if labels:
        labels[-1] = -100  # trailing newline is not part of the loss
    if all(l == -100 for l in labels):
        return None
    return input_ids, labels


_TOKLEN_VERSION = 3  # token-length cache format version (bump when the algorithm changes; old caches auto-invalidate)


def measure_llm_text_lengths(
    tokenizer,
    user_texts,
    answer_texts,
    reasoning_texts=None,
    max_user_tokens: int = 1500,
    max_tokens: int = 4096,
    batch_size: int = 4096,
    user_len_override=None,
) -> list[int]:
    """Measure per sample the "number of rendered text tokens fed to the LLM" (for token-budget
    dynamic batching).

    Approximately exact: the scaffold (chat template + <ts> placeholder wrapper) is rendered once to
    get a constant; user / answer / reasoning are tokenised **in batches** and summed -- an order of
    magnitude faster than per-sample apply_chat_template; the only error comes from BPE merges at
    concatenation boundaries (a few tokens), which is negligible against batch budgets of thousands
    of tokens. The user span is capped at max_user_tokens (middle truncation adds a few marker tokens),
    the whole sequence is capped at max_tokens (over-long samples are skipped and replaced by
    __getitem__; the budget records the maximum).

    ``user_len_override``: per-sample override of the user token count (None entries = tokenise
    normally). Used by ts_as_text (time series rendered as text) -- samples whose number of points
    already far exceeds the user budget necessarily hit ``max_user_tokens``, so simply record the cap
    and **skip the file IO + over-long rendering just to measure the length**.
    """
    user = {"role": "user", "content": wrap_user_with_ts("")}
    asst = {"role": "assistant", "content": ""}
    if reasoning_texts is not None:
        asst["reasoning_content"] = ""
    scaffold = len(tokenizer.apply_chat_template([user, asst], tokenize=True, return_dict=False))

    def _lens(texts):
        out = []
        for s in range(0, len(texts), batch_size):
            enc = tokenizer([str(t) for t in texts[s : s + batch_size]],
                            add_special_tokens=False)["input_ids"]
            out.extend(len(x) for x in enc)
        return out

    if user_len_override is not None:
        # Overridden entries are not tokenised (replaced by an empty string to save time), then the
        # override value takes their place.
        masked = ["" if o is not None else t for t, o in zip(user_texts, user_len_override)]
        u_raw = _lens(masked)
        u = [o if o is not None else l for l, o in zip(u_raw, user_len_override)]
        u = [min(l, max_user_tokens + 8) for l in u]
    else:
        u = [min(l, max_user_tokens + 8) for l in _lens(user_texts)]
    a = _lens(answer_texts)
    if reasoning_texts is not None:
        a = [x + y for x, y in zip(a, _lens(reasoning_texts))]
    return [min(scaffold + uu + aa, max_tokens) for uu, aa in zip(u, a)]


def cached_token_lengths(cache_dir, key_parts, builder) -> list[int]:
    """On-disk cache of token lengths: the key contains the data-file fingerprint (path/mtime/size)
    + truncation parameters + tokenizer + version number; any change rebuilds automatically. With
    cache_dir None the values are computed without caching.

    Multi-process safe: writes go through "temporary file + atomic os.replace" -- np.save straight
    to the final path would let a concurrent rank read a half-written file in the exists->load
    window (with several torchrun ranks sharing the cache directory the first run is necessarily
    concurrent); a corrupted cache (e.g. a previous write killed midway) is treated as a miss and
    rebuilt. On the first run every rank builds once (wasting minutes of tokenisation, but the
    results are identical and later writers overwrite with the same content).

    A failed write is not fatal: the cache only saves tokenisation time; on a full disk / quota
    overrun (ENOSPC inside np.save) warn and
    continue training with the in-memory result."""
    if not cache_dir:
        return builder()
    import hashlib

    import numpy as np

    d = os.path.expanduser(cache_dir)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"toklen_{hashlib.md5(repr(key_parts).encode()).hexdigest()}.npy")
    if os.path.exists(path):
        try:
            return np.load(path).tolist()
        except Exception:
            pass  # half-written / corrupted cache: treat as a miss, rebuild and atomically overwrite
    vals = builder()
    tmp = f"{path}.{os.getpid()}.tmp"
    try:
        with open(tmp, "wb") as f:  # pass a file object: np.save on a bare path forcibly appends .npy
            np.save(f, np.asarray(vals, dtype=np.int64))
        os.replace(tmp, path)
    except OSError as e:
        print(f"[cached_token_lengths] warning: cache write failed ({e}); not persisted, continuing with in-memory result: {path}",
              flush=True)
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return vals


def build_inference_prefix_ids(
    tokenizer,
    user_text: str,
    reasoning: bool = True,
    max_user_tokens: int = 1500,
    max_tokens: int = 4096,
    reasoning_text: str | None = None,
    answer_prefix: str | None = None,
):
    """Inference prefix, **aligned token-for-token** with the training teacher-forcing "conditioning
    context before the answer start".

    Takes LCP(placeholder-answer rendering, empty-answer rendering) -- the same LCP as
    ``build_supervised_ids``, so the prefix necessarily stops right before "the first supervised
    token at training time" and the model continues exactly what it was taught:
    - ``reasoning=False`` (understanding task, empty think): the prefix contains the complete
      ``<think>\\n\\n</think>\\n\\n`` and the model **only continues with the answer**. In
      understanding training this closing scaffold is masked with -100 and never taught, so it must
      go into the prefix rather than be produced by the model.
    - ``reasoning=True`` (forecast task, reasoning inside think): the prefix stops at ``<think>\\n``
      and the model continues with reasoning + ``</think>`` + conclusion (all supervised in training).

    ``reasoning_text`` (meaningful only with ``reasoning=True``) **feeds the ground-truth reasoning
    into the prefix**, extending it past ``<think>ground-truth reasoning</think>`` so the model only
    continues with the answer -- the **teacher-forced diagnostic** of the understanding branch.
    Both renderings fill ``reasoning_content``
    with the **same** ground truth, so the LCP naturally keeps the whole span in the prefix (the
    difference is still only at the answer position); no manual ``</think>`` string concatenation,
    hence no decoupling from how the chat template writes the scaffold.

    ``answer_prefix`` (e.g. ``"Answer:"``) **puts the fixed opening of the answer into the prefix as
    well**, so the model can only continue after it -- "forced answer-prefix decoding": the model
    cannot write a rationale before the answer. It also goes through the LCP: the two renderings fill the answer with
    ``answer_prefix + " x"`` and ``answer_prefix``, so the opening is tokenised **in the same context
    as during training** (no manual token concatenation). Meaningful only when the prefix stops after
    a closed think (``reasoning=False``, or ``reasoning_text`` given); with ``reasoning=True`` and no
    ``reasoning_text`` the prefix stops at the open ``<think>`` and the opening could never appear in
    it, so that silently-ineffective combination raises instead.
    """
    user_text = _truncate_to_tokens(tokenizer, user_text, max_user_tokens)
    user = {"role": "user", "content": wrap_user_with_ts(user_text)}
    full_msg = {"role": "assistant", "content": "x"}
    empty_msg = {"role": "assistant", "content": ""}
    if reasoning:
        r = "x" if reasoning_text is None else reasoning_text
        full_msg["reasoning_content"] = r
        empty_msg["reasoning_content"] = r if reasoning_text is not None else ""
    if answer_prefix:
        if reasoning and reasoning_text is None:
            raise ValueError(
                "answer_prefix is only meaningful when the prefix stops after a closed think "
                "(reasoning=False or reasoning_text given); with reasoning=True the prefix stops at "
                "the open <think> and answer_prefix would never appear in it")
        full_msg["content"] = answer_prefix + " x"
        empty_msg["content"] = answer_prefix
    full = tokenizer.apply_chat_template(
        [user, full_msg], tokenize=True, return_dict=False, truncation=True, max_length=max_tokens
    )
    empty = tokenizer.apply_chat_template(
        [user, empty_msg], tokenize=True, return_dict=False, truncation=True, max_length=max_tokens
    )
    if len(full) >= max_tokens:
        # Tail truncation would cut off the whole assistant header / <think> scaffold and the LCP
        # would degenerate to a truncated user body -- the model would continue a half question
        # instead of answering. Not triggered with the defaults max_user_tokens=1500 << max_tokens=4096.
        import warnings
        warnings.warn(
            f"inference prefix rendering hit the max_tokens={max_tokens} truncation limit; the scaffold may have been cut off; "
            "increase max_tokens or decrease max_user_tokens")
    return list(full[: _lcp_len(full, empty)])


def build_prompt_only_ids(
    tokenizer,
    user_text: str,
    max_user_tokens: int = 1500,
    max_tokens: int = 4096,
):
    """Ablation 3: render only the user turn (including the <ts_start> soft-injection point), with no
    assistant / think / answer.

    The model generates no reasoning/conclusion; only the hidden states of the (plain, trimmed)
    prompt text are fed back into chronos. The caller sets all labels to -100 (no text CE).
    add_generation_prompt=False => a pure user turn, the hidden states contain only the soft prompt
    + the plain prompt text."""
    user_text = _truncate_to_tokens(tokenizer, user_text, max_user_tokens)
    user = {"role": "user", "content": wrap_user_with_ts(user_text)}
    ids = tokenizer.apply_chat_template(
        [user], tokenize=True, return_dict=False, add_generation_prompt=False,
        truncation=True, max_length=max_tokens,
    )
    return list(ids)
