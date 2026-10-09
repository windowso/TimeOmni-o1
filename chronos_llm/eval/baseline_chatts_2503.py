"""Chat-TS (arXiv 2503.10883, Quinlan/Li/Zhu, Queen's University -- not to be confused with ByteDance's
ChatTS 2412.03104 or ChatTime 2412.11376) zero-shot understanding baseline.

## Protocol and deviations

The Chat-TS paradigm discretises the series into tokens and extends the LLM vocabulary so they are mixed
directly into the text stream ("integrates time-series tokens into LLMs' vocabulary"). The paper's final
configuration is Llama-3.1-8B + a simple uniform-bin tokenizer (K=8192, s=±3σ of the z-scored series,
"no training, near-perfect reconstruction").

The released checkpoint (HF PaulQ1/Chat_TS) differs from that configuration:
- its config is the **Llama-2-7B** architecture (rope_theta=1e4 / MHA / 4k ctx; the README's
  "LLama3.1-7B backbone" contradicts the config, and the config is followed);
- vocab_size=41984 = 32000 (Llama) + [PAD] + [SENSOR] + **9982 time-series token slots** -- not the paper's
  8192. 9982 is exactly the paper's formula K-1 bins + 1 channel-end (K=9982 => 9981 bins), and also exactly
  the k9981+1 of the VQVAE checkpoint bundled in the repository. The official preprocessing code is not
  public, so the scheme cannot be confirmed from code.
- This implementation follows the uniform-binning scheme of the Chat-TS paper, with the bin count 9981
  inferred from the released vocabulary (the paper's 8192 pairs with the unreleased Llama-3.1 version),
  s=±3σ, per-channel z-score, channel-end token = last vocabulary slot 41983. Under this scheme the model
  emits coherent English answers, i.e. the token semantics are aligned.
- The context is only 2048 (tokenizer model_max_length; the config says max_position_embeddings=4096 but
  the more conservative 2048 is used) => the series is uniformly resampled down to `--max_ts_tokens`
  (default 1024), leaving room for the question text + generation. Very long series are heavily
  compressed.

## Usage (GPU node; a Python environment with transformers 4.49 is the most stable for Llama-2)

  python chronos_llm/eval/baseline_chatts_2503.py \\
    --model_dir checkpoints/Chat_TS \\
    --jsonl_list chronos_llm/configs/understanding_test_jsonl.txt \\
    --base_dir data/raw/scits/Release_v1 \\
    --out_dir outputs/eval/baseline_chatts_2503/understanding \\
    [--limit 5] [--file_shard_idx 0 --file_num_shards 1] [--prompt_style inst]
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings

import numpy as np
import torch

# ---- vocabulary layout (inferred from the released config/added_tokens, see the module docstring) ----
TEXT_VOCAB = 32002          # 0..31999 Llama + [PAD]=32000 + [SENSOR]=32001
TS_VOCAB = 9982             # 32002..41983
N_BINS = TS_VOCAB - 1       # 9981 numeric bins
CH_END_ID = TEXT_VOCAB + TS_VOCAB - 1   # 41983 = channel-end
S_CLIP = 3.0                # Chat-TS paper: s=±3σ
SENSOR_ID = 32001


def load_ts_first_channel(path, max_len=0):
    """Same loading convention as baseline_timeomnivl_understanding.py."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".npy":
            arr = np.load(path)
        elif ext == ".csv":
            import pandas as pd
            arr = pd.read_csv(path, header=None).apply(
                lambda c: pd.to_numeric(c, errors="coerce")).values
        elif ext in (".wav", ".flac", ".mp3", ".m4a"):
            import soundfile as sf
            arr, _ = sf.read(path, dtype="float32", always_2d=True)
        else:
            raise ValueError(f"unsupported ext {ext}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"load {path} failed: {e}")
        return np.zeros(16, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr.T
    if arr.ndim == 1:
        arr = arr[None, :]
    for row in arr:
        if np.isfinite(row).any():
            s = row
            break
    else:
        return np.zeros(16, dtype=np.float32)
    if max_len and len(s) > max_len:
        idx = np.linspace(0, len(s) - 1, max_len).round().astype(int)
        s = s[idx]
    return np.nan_to_num(s, nan=0.0)


def series_to_token_ids(series_1d: np.ndarray) -> list:
    """Chat-TS binning scheme: per-series z-score -> clip ±3σ -> 9981 uniform bins over [-3,3] -> vocabulary
    offset. A channel-end token is appended (also for a single channel: the Chat-TS paper's "added to mark
    the end of each channel" is unconditional)."""
    x = series_1d.astype(np.float64)
    std = x.std()
    x = (x - x.mean()) / (std if std > 1e-9 else 1.0)
    x = np.clip(x, -S_CLIP, S_CLIP)
    bins = np.floor((x + S_CLIP) / (2 * S_CLIP) * N_BINS).astype(int)
    bins = np.clip(bins, 0, N_BINS - 1)
    ids = (TEXT_VOCAB + bins).tolist()
    ids.append(CH_END_ID)
    return ids


def build_input_ids(tokenizer, question: str, ts_ids: list, prompt_style: str):
    """The text segments are encoded normally by the tokenizer; the time-series token ids are concatenated
    directly (they are not in the text tokenizer's sentencepiece model, so they can only be spliced by id).
    [SENSOR] is used as the leading marker of the time-series block (the official template is not
    public)."""
    if prompt_style == "inst":
        pre = f"[INST] {question}\n\nTime-series data: "
        post = " [/INST]"
    else:  # plain
        pre = f"{question}\n\nTime-series data: "
        post = "\n\nAnswer: "
    pre_ids = tokenizer(pre, add_special_tokens=True)["input_ids"]  # includes <s>
    post_ids = tokenizer(post, add_special_tokens=False)["input_ids"]
    return pre_ids + [SENSOR_ID] + ts_ids + post_ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--jsonl_list", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--max_ts_tokens", type=int, default=1024,
                     help="resampling cap for the series (ctx 2048 - text - generation headroom)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--exclude", default="MIMII")
    ap.add_argument("--prompt_style", default="inst", choices=["inst", "plain"])
    ap.add_argument("--file_shard_idx", type=int, default=0)
    ap.add_argument("--file_num_shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[chatts] loading {args.model_dir} ...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()
    print(f"[chatts] loaded in {time.time()-t0:.1f}s; vocab={model.config.vocab_size}")
    assert model.config.vocab_size == TEXT_VOCAB + TS_VOCAB, \
        f"vocabulary layout assumption violated: {model.config.vocab_size} != {TEXT_VOCAB + TS_VOCAB}"

    files = [l.strip() for l in open(args.jsonl_list)
             if l.strip() and not l.strip().startswith("#")]
    files = [f for f in files if not any(e in f for e in args.exclude.split(",") if e)]
    if args.file_num_shards > 1:
        files = files[args.file_shard_idx:: args.file_num_shards]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[chatts] shard {args.file_shard_idx}/{args.file_num_shards}: {len(files)} files")

    eos_id = tokenizer.eos_token_id
    for jf in files:
        rows = [json.loads(l) for l in open(jf)]
        if args.limit:
            rows = rows[: args.limit]
        name = os.path.basename(jf)
        out_path = os.path.join(args.out_dir, name)
        if os.path.exists(out_path):
            print(f"[skip] {name} already exists")
            continue
        print(f"[chatts] {name}: {len(rows)} rows")
        results = []
        n_fail = 0
        for ridx, r in enumerate(rows):
            t_row = time.time()
            ori = ((r.get("input_ts") or {}).get("original") or {}).get("ori_path") or ""
            path = ori if os.path.isabs(ori) else os.path.join(args.base_dir, ori)
            q = (r.get("input_text") or [""])[0]
            gen_text = ""
            try:
                s = load_ts_first_channel(path, max_len=args.max_ts_tokens)
                ts_ids = series_to_token_ids(s)
                input_ids = build_input_ids(tokenizer, q, ts_ids, args.prompt_style)
                # Fallback truncation: when text + series exceed 2048 - budget, cut the middle of the sequence (keep head and tail)
                budget = 2048 - args.max_new_tokens - 8
                if len(input_ids) > budget:
                    overflow = len(input_ids) - budget
                    mid = len(input_ids) // 2
                    input_ids = input_ids[: mid - overflow // 2 - overflow % 2] + input_ids[mid + overflow // 2:]
                ids = torch.tensor([input_ids], device="cuda:0")
                with torch.no_grad():
                    out = model.generate(
                        ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                        eos_token_id=eos_id, pad_token_id=eos_id)
                gen_ids = out[0][ids.shape[1]:]
                # The model may emit time-series tokens (id >= 32002, outside the text tokenizer's range;
                # decode would crash / produce garbage): filter them out before decoding and record their
                # share as a diagnostic (many TS tokens => degenerate output)
                gen_list = gen_ids.tolist()
                n_ts = sum(1 for t in gen_list if t >= TEXT_VOCAB)
                text_ids = [t for t in gen_list if t < TEXT_VOCAB]
                gen_text = tokenizer.decode(text_ids, skip_special_tokens=True).strip()
                if n_ts:
                    gen_text = f"[TS_TOKENS x{n_ts}] " + gen_text
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                print(f"  [warn] {name} row {ridx} failed: {type(e).__name__}: {e}")

            results.append({
                "id": r.get("id"), "uid": r.get("uid"),
                "dataset_name": r.get("dataset_name"), "task": r.get("task"),
                "scene": r.get("scene"), "input_text": r.get("input_text"),
                "generated_text": gen_text,
                "ground_truth": r.get("gt_text"), "gt_result": r.get("gt_result"),
            })
            if (ridx + 1) % 20 == 0 or ridx == 0:
                print(f"  {ridx+1}/{len(rows)} (last row {time.time()-t_row:.1f}s, failures so far {n_fail})")

        with open(out_path, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  -> {out_path} (fail {n_fail}/{len(rows)})")


if __name__ == "__main__":
    main()
