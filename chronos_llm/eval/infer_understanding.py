"""Understanding-task inference: run generate_understanding on every jsonl and write the text to
JSONL (schema aligned with TimeOmni's infer_benchmark: id/dataset_name/task/scene/generated_text/
gt_result/ground_truth, consumed by eval_understanding for the metrics; additionally gt_reasoning --
the reference reasoning text when the sample carries a think field, used by LLM-judge style
reasoning-quality evaluations to compare against the <think> segment of the generated text; the
empty string for samples without think).

``run_understanding_infer`` takes an already loaded model (decoupled from loading, so a tiny model
can be used for a CPU smoke test); ``main`` calls it after ``from_pretrained``. Supports torchrun
multi-node multi-GPU: each rank takes a strided shard of the samples (every sample exactly once),
writes a ``.rank{r}`` shard file, and after a barrier rank0 merges them in the original row order
into the final file; single-process behaviour is unchanged.
See chronos_llm/scripts/eval_understanding.sh for usage.
"""
import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from chronos_llm.data.collator import ChronosLLMCollator
from chronos_llm.data.understanding_dataset import UnderstandingJsonlDataset
from chronos_llm.eval.dist_utils import barrier, dist_info, init_distributed
from chronos_llm.models.chronos_llm_model import ChronosLLM


def _to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _merge_jsonl_shards(out_file, world):
    """Merge the per-rank shards (rows carry the original row number in ``_idx``), restore the original
    row order, write the final file and remove the shards."""
    merged = []
    for r in range(world):
        part = f"{out_file}.rank{r}"
        with open(part, encoding="utf-8") as f:
            merged.extend(json.loads(line) for line in f if line.strip())
        os.remove(part)
    merged.sort(key=lambda x: x["_idx"])
    for m in merged:
        m.pop("_idx")
    _write_jsonl(out_file, merged)


def _limit_indices(total, limit):
    """Sample indices of the limited subset: evenly strided over the whole file, order-preserving, deterministic.

    Returns all indices when ``limit<=0`` or ``limit>=total``.
    """
    if limit <= 0 or limit >= total:
        return list(range(total))
    return [(i * total) // limit for i in range(limit)]


def run_understanding_infer(model, tok, data_files, output_folder, *, base_dir=None,
                            batch_size=8, max_new_tokens=128, device="cuda",
                            max_user_tokens=1500, max_tokens=4096, num_workers=4, limit=0,
                            sample_chunks=0, overview_chunk=True, sample_window=8192,
                            no_reasoning=False, teacher_forced_reasoning=False,
                            force_reasoning=False, counterfactual_reasoning=False,
                            ts_as_text=False, answer_prefix=None):
    """Generate text file by file; every input jsonl gets an output JSONL of the same name. Returns the
    list of output files.

    ``answer_prefix`` (e.g. ``"Answer:"``): forced answer-prefix decoding -- the prefix already
    contains the opening of the answer and the model only continues after it; the ``generated_text``
    that is written out **prepends the opening again**, so the extractors (which look for a leading
    ``Answer:``) stay on the same axis as every other protocol.

    Under torchrun multi-processing the samples are automatically sharded with a stride
    (``range(rank, N, world)``: every sample exactly once, balanced load); each rank writes a
    ``.rank{r}`` shard and rank0 merges them after the barrier; a single process writes the final file
    directly, behaviour unchanged.
    With ``limit>0`` only limit samples per file are evaluated, chosen by **evenly strided sampling over
    the whole file** (``int(i*len/limit)``) and **not the first limit rows** -- the jsonl files produced
    by the conversion scripts are often laid out in subtask blocks (e.g. HiTSR L2 then L3), so taking
    the head would cover only the first subtask. The strided subset is **deterministic**.
    """
    rank, world = dist_info()
    os.makedirs(output_folder, exist_ok=True)
    out_files = []
    for data_file in data_files:
        ds = UnderstandingJsonlDataset([data_file], tok, base_dir=base_dir, inference_mode=True,
                                       max_user_tokens=max_user_tokens, max_tokens=max_tokens,
                                       sample_chunks=sample_chunks, overview_chunk=overview_chunk,
                                       sample_window=sample_window, no_reasoning=no_reasoning,
                                       teacher_forced_reasoning=teacher_forced_reasoning,
                                       force_reasoning=force_reasoning,
                                       counterfactual_reasoning=counterfactual_reasoning,
                                       ts_as_text=ts_as_text, answer_prefix=answer_prefix)
        idx = _limit_indices(len(ds), limit)
        n = len(idx)
        shard = [idx[i] for i in range(rank, n, world)]
        dl = DataLoader(Subset(ds, shard), batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=ChronosLLMCollator(tokenizer=tok))
        if rank == 0:
            print(f"\n{os.path.basename(data_file)}: {n} samples"
                  + (f" (sharded over {world} ranks)" if world > 1 else ""))
        results = []
        for batch in tqdm(dl, desc=os.path.basename(data_file), disable=rank != 0):
            metas = batch["meta"]
            with torch.no_grad():
                texts = model.generate_understanding(_to_device(batch, device),
                                                     max_new_tokens=max_new_tokens, do_sample=False)
            for m, t in zip(metas, texts):
                results.append({
                    "id": m.get("id"), "uid": m.get("uid"),
                    "dataset_name": m.get("dataset_name"), "task": m.get("task"),
                    "scene": m.get("scene"), "input_text": m.get("input_text"),
                    # answer_prefix sits in the prefix, not in the generation => prepend it back so
                    # the extractors can find the answer by the leading "Answer:"
                    "generated_text": ((answer_prefix or "") + (t or "")).strip(),
                    "ground_truth": m.get("ground_truth"), "gt_result": m.get("gt_result"),
                    "gt_reasoning": m.get("gt_reasoning", ""),
                })
        out_file = os.path.join(output_folder, os.path.basename(data_file))
        if world > 1:
            for j, row in enumerate(results):
                row["_idx"] = shard[j]
            _write_jsonl(f"{out_file}.rank{rank}", results)
            barrier()  # wait until every rank's shard is on disk before rank0 merges
            if rank == 0:
                _merge_jsonl_shards(out_file, world)
        else:
            _write_jsonl(out_file, results)
        if rank == 0:
            print(f"  -> {out_file}")
        out_files.append(out_file)
    return out_files


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_file_list", nargs="+", required=True, help="list of understanding test jsonl files")
    ap.add_argument("--output_folder", required=True)
    ap.add_argument("--base_dir", default=None, help="prefix for relative time-series paths (ori_path)")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--max_user_tokens", type=int, default=1500)
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--encode_sample_chunks", type=int, default=0,
                    help="number of windows K for uniform window sampling of over-long histories (must match training; 0=off, truncate instead)")
    ap.add_argument("--encode_overview_chunk", type=int, default=1,
                    help="whether to append a downsampled whole-series overview chunk when sampling triggers (1/0, must match training)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0,
                    help="evaluate only N samples per test jsonl (0=all) -- a deterministic subset for quick comparisons")
    ap.add_argument("--no_reasoning", action="store_true",
                    help="evaluate a no-CoT control model: the inference prefix contains a complete empty `<think></think>` and the model only continues with the answer."
                         " **Must match the checkpoint's training setting** -- evaluating a no-CoT model with a CoT prefix"
                         " (or vice versa) makes it continue a segment it never learned, distorting the metrics.")
    ap.add_argument("--teacher_forced_reasoning", action="store_true",
                    help="diagnostic: feed the sample's reference reasoning into the prefix so the model only"
                         " continues with the answer.")
    ap.add_argument("--force_reasoning", action="store_true",
                    help="always stop the inference prefix at the opening `<think>` and let the model write its own reasoning, regardless of whether"
                         " the test sample carries a think field. Use it for a model trained with reasoning: such a model never learned to"
                         " answer directly, so test sets without a think field would otherwise go through a prefix it never learned."
                         " Mutually exclusive with --no_reasoning.")
    ap.add_argument("--counterfactual_reasoning", action="store_true",
                    help="diagnostic: the prefix carries the reasoning of another sample from the same task (cyclic shift)."
                         " Takes precedence when enabled together with --teacher_forced_reasoning.")
    ap.add_argument("--answer_prefix", default=None,
                    help="diagnostic: put the fixed opening of the answer (e.g. 'Answer:') into the prefix right after the"
                         " closed think, so the model can only fill in the answer. Must be combined with --no_reasoning"
                         " (a test set with think raises otherwise); the generated_text that is written out prepends the"
                         " opening again.")
    ap.add_argument("--llm_only_zeroshot", action="store_true",
                    help="LLM-only **zero-shot**: --model_path points directly at a bare LLM directory (e.g. the Qwen3.5-9B"
                         " checkpoint, not a ChronosLLM save directory); the model is cold-assembled with the llm_only config (no"
                         " chronos/LoRA) and the time series is textualised at full precision into the prompt. A fine-tuned llm_only"
                         " checkpoint does not need this flag (its config.json carries llm_only and goes through from_pretrained).")
    args = ap.parse_args(argv)

    rank, world, local_rank = init_distributed()
    if world > 1 and args.device == "cuda":
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    if rank == 0:
        print(f"loading model {args.model_path} ...")
    if args.llm_only_zeroshot:
        from chronos_llm.models.chronos_llm_model import ChronosLLMConfig

        model = ChronosLLM.from_config(
            ChronosLLMConfig(llm_only=True, llm_path=args.model_path)).eval().to(args.device)
    else:
        model = ChronosLLM.from_pretrained(args.model_path, merge=True).eval().to(args.device)
    # llm_only (zero-shot or fine-tuned checkpoint) => the evaluation data automatically switches to ts_as_text, consistent with training.
    ts_as_text = bool(getattr(model.config, "llm_only", False))
    run_understanding_infer(model, model.tokenizer, args.data_file_list, args.output_folder,
                            base_dir=args.base_dir, batch_size=args.batch_size,
                            max_new_tokens=args.max_new_tokens, device=args.device,
                            max_user_tokens=args.max_user_tokens, max_tokens=args.max_tokens,
                            num_workers=args.num_workers, limit=args.limit,
                            sample_chunks=args.encode_sample_chunks,
                            overview_chunk=bool(args.encode_overview_chunk),
                            no_reasoning=args.no_reasoning,
                            teacher_forced_reasoning=args.teacher_forced_reasoning,
                            force_reasoning=args.force_reasoning,
                            counterfactual_reasoning=args.counterfactual_reasoning,
                            ts_as_text=ts_as_text, answer_prefix=args.answer_prefix)


if __name__ == "__main__":
    main()
