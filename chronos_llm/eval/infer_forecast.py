"""Forecast-task inference: run generate_forecast on the parquet split=test. **Numbers** (quantile
predictions/gt/roi/valid/quantile_levels/ids) go to an npz; **text** goes to a companion
human-readable jsonl (same stem as the npz, suffix .jsonl, one line per sample with
id/prompt/gen_text/gt_reasoning/gt_conclusion/median_pred/gt_series/roi, arrays trimmed to the valid
horizon, values rounded to 4 decimals). Text is not stored in the npz.
The npz is the numeric source for metrics (eval_forecast.py); the jsonl serves for inspecting
"generated reasoning/conclusion vs. ground-truth text" and "median prediction vs. numeric ground
truth". Note that ``gt_series`` is the **numeric time
series ground truth** (a different meaning from the textual ``ground_truth`` of the understanding
task, hence the distinct name).

Two-stage design (following TimeOmni infer/eval): this script only produces predictions; metrics are
computed by eval_forecast.py. ``run_forecast_infer`` takes an already-loaded model (decoupled from
loading so a tiny model can smoke-test it on CPU); ``main`` calls it after ``from_pretrained``.
Supports torchrun multi-node/multi-GPU: each rank takes a strided shard of samples, writes a
``.rank{r}.npz`` shard, and after a barrier rank 0 merges them into the final npz (the jsonl is
sharded per rank and merged by rank 0 the same way); single-process behaviour is unchanged.
Usage: see chronos_llm/scripts/eval_forecast.sh.
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.forecast_dataset import ForecastParquetDataset
from chronos_llm.eval.dist_utils import barrier, dist_info, init_distributed
from chronos_llm.eval.metrics import _median_index
from chronos_llm.models.chronos_llm_model import ChronosLLM


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _jsonl_path(output):
    """npz output path -> companion jsonl path (.npz replaced by .jsonl, otherwise .jsonl appended)."""
    return output[:-4] + ".jsonl" if output.endswith(".npz") else output + ".jsonl"


def _forecast_jsonl_rows(ids, metas, texts, preds, gts, rois, valids, levels):
    """Build human-readable jsonl rows from the per-sample raw (un-padded) arrays + per-row meta:

    - ``gen_text`` model-generated reasoning/conclusion vs. ``gt_reasoning``/``gt_conclusion`` text truth;
    - ``median_pred`` predicted median vs. ``gt_series`` numeric truth (both trimmed to the valid
      horizon and rounded to 4 decimals);
    - ``roi`` converted to a future-relative interval ``[start, end)`` (null when there is no ROI).
    Teacher-forced mode has no meta (meta={}) => prompt/gt_* are empty strings.
    """
    qi = _median_index(levels)
    rows = []
    for i in range(len(ids)):
        m = metas[i]
        valid = np.asarray(valids[i], dtype=bool)
        h = int(valid.sum())                                   # valid horizon (fully observed future => =future_len)
        med = np.asarray(preds[i])[qi, :h].astype(float)
        gt = np.asarray(gts[i], dtype=float)[:h]
        roi_where = np.where(np.asarray(rois[i], dtype=float)[:h] > 0.5)[0]
        roi_span = [int(roi_where[0]), int(roi_where[-1] + 1)] if roi_where.size else None
        rows.append({
            "id": ids[i],
            "prompt": m.get("input_text", ""),
            "gen_text": texts[i],
            "gt_reasoning": m.get("gt_reasoning", ""),
            "gt_conclusion": m.get("gt_conclusion", ""),
            "median_pred": [round(float(x), 4) for x in med],
            "gt_series": [round(float(x), 4) for x in gt],
            "roi": roi_span,
        })
    return rows


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _merge_jsonl_shards(jsonl, world):
    """Merge the per-rank jsonl shards (each row carries its original row index in ``_idx``),
    restore the original row order, write the final file and remove the shards."""
    merged = []
    for r in range(world):
        part = f"{jsonl}.rank{r}"
        with open(part, encoding="utf-8") as f:
            merged.extend(json.loads(line) for line in f if line.strip())
        os.remove(part)
    merged.sort(key=lambda x: x["_idx"])   # stable: multiple target rows of one sample keep their append order
    for m in merged:
        m.pop("_idx")
    _write_jsonl(jsonl, merged)


def run_forecast_infer(model, tok, dataset, output, *, batch_size=8, max_new_tokens=256,
                       device="cuda", num_workers=4, teacher_forced=False, num_beams=1,
                       ensemble_k=1, fb_mask_generated=False):
    """Run forecasting on a forecast dataset and save an npz; returns the aligned per-sample arrays.

    ``teacher_forced=True`` (exposure-bias control): the dataset must be the **training rendering**
    (inference_mode=False, input_ids containing the true reasoning/conclusion); instead of
    autoregressive generation, a teacher-forced forward pass provides the hidden states fed back into
    chronos. Together with the default mode (self-generated reasoning) this yields two npz files; the
    difference between their eval metrics is the exposure bias.

    Under torchrun with several processes, samples are sharded with a stride (each sample exactly
    once); each rank saves ``.rank{r}.npz``, and after a barrier rank 0 merges them (re-padding to the
    global max FL and sorting back to the original row order), writes the final npz and returns it;
    non-zero ranks return None.
    """
    rank, world = dist_info()
    # This function is responsible for creating the output's parent directory: the forecast-only
    # path of mid_eval has no makedirs upstream.
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if teacher_forced and num_workers:
        # The lazy `_skip` happens inside the DataLoader worker's copy of the dataset and is never
        # written back to the main process -- the row-alignment guard (the inner._skip check below)
        # is always empty with num_workers>0 and therefore useless. Row alignment is a hard
        # constraint in teacher-forced mode, so force single-process loading to make the guard real.
        num_workers = 0
    if getattr(model, "chronos", None) is not None:
        levels = model.chronos.quantiles.detach().cpu().float().numpy()  # (Q,)
    else:
        # llm_only: no chronos; the point forecast is copied to the same 21 Chronos-2 quantile
        # levels.
        from chronos_llm.data.ts_text import CHRONOS2_QUANTILE_LEVELS

        levels = np.asarray(CHRONOS2_QUANTILE_LEVELS, dtype=np.float32)
    shard = list(range(rank, len(dataset), world))
    if world > 1:
        dataset = Subset(dataset, shard)
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    collate_fn=ChronosLLMCollator(tokenizer=tok))
    preds, gts, rois, valids, ids, row_orig, texts, metas = [], [], [], [], [], [], [], []
    cursor = 0  # sample cursor within this rank's shard (dl consumes the shard in order, shuffle=False)
    for batch in tqdm(dl, desc="forecast infer" + (" (teacher-forced)" if teacher_forced else ""),
                      disable=rank != 0):
        # preds/gt/roi are expanded per target row (sum of n_targets); meta is per sample -> replicate
        # it n_targets times to row level so that ids of multi-target samples align with the rows.
        n_tg = batch.get("n_targets")
        counts = n_tg.tolist() if n_tg is not None else [1] * batch["future"].shape[0]
        meta_samples = batch.get("meta", [{}] * len(counts))
        meta = [m for m, c in zip(meta_samples, counts) for _ in range(c)]
        # orig_idx must likewise be **row-level** (same length as preds/ids): once a multi-target
        # sample is expanded into several rows, a sample-level orig_idx would make the merge-side
        # argsort length disagree with the row count -- rows silently dropped and predictions/truth
        # misaligned. Rows of one sample share its sample index; the merge side's stable argsort
        # keeps the within-sample order.
        batch_orig = shard[cursor:cursor + len(counts)]
        cursor += len(counts)
        row_orig.extend(o for o, c in zip(batch_orig, counts) for _ in range(c))
        gt, roi = batch["future"], batch["roi_mask"]   # (sum n_targets, FL): NaN pad / {0,1}
        fl = gt.shape[-1]
        with torch.no_grad():
            if teacher_forced:
                out = model.generate_forecast_teacher_forced(_to_device(batch, device), horizon=fl)
            else:
                out = model.generate_forecast(_to_device(batch, device), horizon=fl,
                                              max_new_tokens=max_new_tokens, do_sample=False,
                                              ensemble_k=ensemble_k,
                                              fb_mask_generated=fb_mask_generated,
                                              **({"num_beams": num_beams} if num_beams > 1 else {}))
        qp = out["quantile_preds"].detach().cpu().float().numpy()[..., :fl]  # (B, Q, FL)
        gt_np, roi_np = gt.numpy(), roi.numpy()
        valid = ~np.isnan(gt_np)
        # Model-generated reasoning text (generate mode only; teacher-forced generates nothing ->
        # empty string). out["text"] is per **sample** (one generation per soft prompt) -> expand by
        # n_targets to **row level**, same length as preds/ids (rows of a multi-target sample share
        # that sample's generated text).
        if teacher_forced:
            texts_batch = [""] * gt_np.shape[0]
        else:
            texts_batch = [t for t, c in zip(out["text"], counts) for _ in range(c)]
        for i in range(gt_np.shape[0]):
            preds.append(qp[i]); gts.append(gt_np[i]); rois.append(roi_np[i])
            valids.append(valid[i]); ids.append(meta[i].get("id", str(len(ids))))
            texts.append(texts_batch[i]); metas.append(meta[i])

    # Teacher-forced uses the training rendering: if any sample was lazily skipped/replaced for
    # being over-long, rows no longer correspond one-to-one with inference mode.
    inner = dataset
    while hasattr(inner, "dataset"):  # unwrap nested Subsets (limit truncation + rank sharding)
        inner = inner.dataset
    if teacher_forced and getattr(inner, "_skip", None):
        import warnings
        warnings.warn(f"teacher-forced mode: {len(inner._skip)} samples were skipped/replaced for being "
                      f"over-long; row alignment with inference mode is broken")

    # FL may differ per sample -> right-pad to this rank's max with NaN/False and store a valid mask
    # (re-padded to the global max at merge time).
    Q, N = levels.shape[0], len(preds)
    Tmax = max((p.shape[-1] for p in preds), default=1)
    P = np.full((N, Q, Tmax), np.nan, np.float32)
    G = np.full((N, Tmax), np.nan, np.float32)
    R = np.zeros((N, Tmax), np.float32)
    V = np.zeros((N, Tmax), bool)
    for i in range(N):
        t = preds[i].shape[-1]
        P[i, :, :t] = preds[i]; G[i, :t] = gts[i]; R[i, :t] = rois[i]; V[i, :t] = valids[i]
    # Per-row dataset_name (from meta; the training rendering also has it with emit_meta=True) --
    # stored directly in the npz so per-dataset analyses group by the npz's own labels instead of
    # relying on "positional alignment with the test split" (ids are not unique + the tf training
    # rendering filters empty-conclusion rows -> row order may differ from the test split, and
    # positional alignment would cross wires / break).
    ds_names = [str(m.get("dataset_name", "")) for m in metas]
    # Human-readable jsonl rows (from the per-sample un-padded arrays, trimmed to the valid horizon);
    # written alongside the npz.
    rows = _forecast_jsonl_rows(ids, metas, texts, preds, gts, rois, valids, levels)
    jsonl = _jsonl_path(output)
    # gen_text and other text go only to the jsonl, not into the npz.
    if world == 1:
        np.savez_compressed(output, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
                            quantile_levels=levels, ids=np.array(ids, dtype=object),
                            dataset_names=np.array(ds_names, dtype=object))
        _write_jsonl(jsonl, rows)
        print(f"Saved {N} predictions -> {output} / {jsonl}")
        return dict(pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V, quantile_levels=levels)
    np.savez_compressed(f"{output}.rank{rank}.npz", pred_quantiles=P, gt=G, roi_mask=R,
                        valid_mask=V, quantile_levels=levels, ids=np.array(ids, dtype=object),
                        dataset_names=np.array(ds_names, dtype=object),
                        orig_idx=np.array(row_orig, np.int64))
    for r, oi in zip(rows, row_orig):   # jsonl shards carry the original row index so rank 0 can restore the order
        r["_idx"] = int(oi)
    _write_jsonl(f"{jsonl}.rank{rank}", rows)
    barrier()  # wait until every rank's shard is on disk before rank 0 merges
    if rank != 0:
        return None
    out = _merge_npz_shards(output, world)
    _merge_jsonl_shards(jsonl, world)
    print(f"Saved {out['gt'].shape[0]} predictions -> {output} / {jsonl}")
    return out


def _merge_npz_shards(output, world):
    """Merge the per-rank npz shards: re-pad to the global max FL, restore the original row order by
    orig_idx, write the final npz and remove the shards."""
    parts = []
    for r in range(world):
        part = f"{output}.rank{r}.npz"
        with np.load(part, allow_pickle=True) as z:
            parts.append({k: z[k] for k in z.files})
        os.remove(part)
    levels = parts[0]["quantile_levels"]
    Q = levels.shape[0]
    Tg = max(int(p["gt"].shape[-1]) for p in parts)
    Ng = sum(int(p["gt"].shape[0]) for p in parts)
    P = np.full((Ng, Q, Tg), np.nan, np.float32)
    G = np.full((Ng, Tg), np.nan, np.float32)
    R = np.zeros((Ng, Tg), np.float32)
    V = np.zeros((Ng, Tg), bool)
    ids, dsn, order, row = [], [], [], 0
    for p in parts:
        n, t = p["gt"].shape
        P[row:row + n, :, :t] = p["pred_quantiles"]
        G[row:row + n, :t] = p["gt"]
        R[row:row + n, :t] = p["roi_mask"]
        V[row:row + n, :t] = p["valid_mask"]
        ids.extend(p["ids"].tolist()); order.extend(p["orig_idx"].tolist())
        if "dataset_names" in p:  # backward compatibility with old shards (empty string if the key is absent)
            dsn.extend(p["dataset_names"].tolist())
        else:
            dsn.extend([""] * n)
        row += n
    # orig_idx is a row-level sample index (rows of a multi-target sample share one value): the stable
    # sort keeps each sample's rows in their within-shard append order == original within-sample
    # order. (gen_text and other text go through the jsonl shard merge, not the npz.)
    sortidx = np.argsort(np.array(order), kind="stable")
    P, G, R, V = P[sortidx], G[sortidx], R[sortidx], V[sortidx]
    np.savez_compressed(output, pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V,
                        quantile_levels=levels,
                        ids=np.array([ids[i] for i in sortidx], dtype=object),
                        dataset_names=np.array([dsn[i] for i in sortidx], dtype=object))
    return dict(pred_quantiles=P, gt=G, roi_mask=R, valid_mask=V, quantile_levels=levels)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="directory to load with from_pretrained (adapter + config)")
    ap.add_argument("--parquet", required=True,
                    help="forecast parquet path (passed by the calling script rather than hard-coded, "
                         "so the dataset is easy to swap)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=320,
                    help="covers the generated segment (reasoning+</think>+conclusion+eos)")
    ap.add_argument("--max_user_tokens", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--num_beams", type=int, default=1,
                    help=">1 uses beam search in generate (mitigates greedy sub-optimality)")
    ap.add_argument("--ensemble_k", type=int, default=1,
                    help=">1 enables the K-sample feedback ensemble: greedy + K-1 sampled conclusions "
                         "are each fed back, and the quantile curves are averaged point-wise "
                         "(vincentization)")
    ap.add_argument("--fb_mask_generated", action="store_true",
                    help="diagnostic: mask the generated segment in the feedback and use only the "
                         "prompt-prefix hidden states")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no_merge", action="store_true",
                    help="do not merge LoRA; keep the adapter applied on the fly (PeftModel forward, no "
                         "merge_and_unload), the same inference path as the in-training mid_eval.")
    ap.add_argument("--gate_scale", type=float, default=1.0,
                    help="diagnostic: after loading, multiply the gate parameters of chronos' gated "
                         "cross-attn by a factor alpha (default 1 = unchanged)")
    ap.add_argument("--output", required=True,
                    help="output .npz for generate mode (or for teacher_forced when mode=teacher_forced)")
    ap.add_argument("--mode", default="both", choices=["generate", "teacher_forced", "both"],
                    help="generate = autoregressively generate the reasoning; teacher_forced = true "
                         "reasoning control (exposure bias); both = load the model once and run both "
                         "modes (the eval-metric difference between the two npz files is the exposure bias)")
    ap.add_argument("--tf_output", default=None,
                    help="teacher-forced output .npz when mode=both; defaults to --output with suffix _tf")
    ap.add_argument("--llm_only_zeroshot", action="store_true",
                    help="LLM-only **zero-shot**: --model_path points directly at a bare LLM directory, "
                         "cold-assembled with the llm_only config (no chronos/LoRA); the history is "
                         "textualized at full precision into the prompt and numbers are parsed from the "
                         "generated text. Fine-tuned llm_only checkpoints do not need this switch "
                         "(their config.json already carries llm_only).")
    args = ap.parse_args(argv)

    rank, world, local_rank = init_distributed()
    if world > 1 and args.device == "cuda":
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    if rank == 0:
        print(f"Loading model {args.model_path} ...")
    if args.llm_only_zeroshot:
        from chronos_llm.models.chronos_llm_model import ChronosLLMConfig

        model = ChronosLLM.from_config(
            ChronosLLMConfig(llm_only=True, llm_path=args.model_path)).eval().to(args.device)
    else:
        model = ChronosLLM.from_pretrained(args.model_path, merge=not args.no_merge).eval().to(args.device)
    if getattr(model.config, "llm_only", False) and args.mode != "generate":
        if rank == 0:
            print("[infer_forecast] llm_only has no teacher-forced protocol; forcing mode=generate")
        args.mode = "generate"
    if args.gate_scale != 1.0:
        # The gate is used through tanh; for small g, tanh(alpha*g) ~ alpha*tanh(g), i.e. the
        # LLM->chronos feedback gain is scaled by alpha, touching only the gate scalars.
        n = 0
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "chronos" in name and "cross_attn.gate" in name:
                    p.mul_(args.gate_scale)
                    n += 1
        if rank == 0:
            print(f"[gate_scale] x{args.gate_scale} applied to {n} chronos cross_attn.gate parameters")
    tok = model.tokenizer
    # prompt_only (no-reasoning) checkpoints: evaluation must also use plain-prompt-only ids
    # (otherwise generate would produce a reasoning scaffold the model never saw in training) --
    # read from the model config, same protocol as mid_eval.py.
    prompt_only = bool(getattr(model.config, "forecast_prompt_only", False))

    def _make_ds(inference_mode):
        # teacher-forced uses the training rendering (input_ids with the true reasoning/conclusion);
        # generate uses the inference prefix.
        ds = ForecastParquetDataset(args.parquet, tok, split=args.split,
                                    inference_mode=inference_mode,
                                    forecast_prompt_only=prompt_only,
                                    ts_as_text=bool(getattr(model.config, "llm_only", False)),
                                    max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens,
                                    emit_meta=True)  # tf (training rendering) also carries real id/dataset_name
        return ds, len(ds)

    runs = []  # (teacher_forced, output)
    if args.mode in ("generate", "both"):
        runs.append((False, args.output))
    if args.mode in ("teacher_forced", "both"):
        tf_out = args.output if args.mode == "teacher_forced" else (
            args.tf_output or (args.output[:-4] + "_tf.npz" if args.output.endswith(".npz")
                               else args.output + "_tf")
        )
        runs.append((True, tf_out))

    n_inf = None
    for teacher_forced, output in runs:
        ds, n_full = _make_ds(inference_mode=not teacher_forced)
        if teacher_forced:
            # The rows of the two modes must correspond one-to-one (aligned by row order): the training
            # rendering filters out empty conclusions, so if the row count differs from inference mode
            # the two npz files are not comparable -- fail explicitly.
            if n_inf is None:
                n_inf = _make_ds(inference_mode=True)[1]
            assert n_full == n_inf, (
                f"teacher-forced row count {n_full} != inference mode {n_inf} (empty-conclusion rows "
                f"were filtered); the two result sets cannot be aligned by row"
            )
        else:
            n_inf = n_full
        if args.limit and args.limit < len(ds):
            ds = Subset(ds, range(args.limit))
        if rank == 0:
            print(f"{args.split} split size = {len(ds)} ({'teacher-forced' if teacher_forced else 'generate'})")
        run_forecast_infer(model, tok, ds, output, batch_size=args.batch_size,
                           max_new_tokens=args.max_new_tokens, device=args.device,
                           num_workers=args.num_workers, teacher_forced=teacher_forced,
                           num_beams=args.num_beams, ensemble_k=args.ensemble_k,
                           fb_mask_generated=args.fb_mask_generated)


if __name__ == "__main__":
    main()
