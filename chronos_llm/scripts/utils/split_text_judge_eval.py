r"""Forecasting-side consistency judge with split evaluation and mismatched-partner pairing.

1. **Split evaluation**: the generation is split at `</think>` into reasoning and conclusion, and
   each part is judged against its own reference:
   - **conclusion judge**: `gen_conclusion` vs `gt_conclusion`, reusing the
     `text_alignment_eval.SHAPE_JUDGE_*` prompts unchanged;
   - **reasoning judge**: `gen_reasoning` vs `gt_reasoning`, with a prompt defined in this module
     whose scale mirrors the shape judge's.
   Consistency = the per-sample mean of the conclusion and reasoning scores.
2. **Mismatched partner**: each sample is also judged against the reference of another sample
   (`build_pairing`; a derangement of the whole pool for full runs), and the same-dataset rate of the
   partners is reported.

Four conditions = {conclusion, reasoning} x {matched, mismatched}: n samples => 4n calls.
`--merged` adds a judge of the whole generation against the whole reference.

The judge API is reached over the public internet, so run this from a machine with outbound network
access.

`random_partners` derives its mapping from `sel` itself (one partner per entry, in `sel` order), so
changing `--n` requires changing `--out` as well; otherwise the partners stored under `*|shuf|i` in
the old checkpoint file no longer match the new mapping. `*|real|*` entries are unaffected.

    python chronos_llm/scripts/utils/split_text_judge_eval.py \
        <prediction dir> --n <number of test rows> --out outputs/eval/split_judge
"""
import argparse
import concurrent.futures
import json
import os
import random

from chronos_llm.eval.text_alignment_eval import (
    DEFAULT_JUDGE_MODEL,
    SHAPE_JUDGE_SYSTEM,
    SHAPE_JUDGE_USER_TEMPLATE,
    _retry_exceptions,
    _with_retry,
    build_openai_client,
    domain_of,
    parse_judge_response,
)

REASONING_JUDGE_SYSTEM = (
    "You are a careful evaluator comparing two analytical reasoning passages written about "
    "the same time-series forecasting task. Judge ONLY whether the model's reasoning reaches "
    "the same substantive analysis as the reference: which part of the future window is "
    "identified as the region of interest, how the stated event is expected to affect it, and "
    "what behaviour the history is said to support. Explicitly IGNORE writing style, ordering, "
    "verbosity, and exact clock times / index numbers."
)

REASONING_JUDGE_USER_TEMPLATE = """Compare the substantive analysis in the two passages below.

[Model-generated reasoning]
{gen_text}

[Reference (ground-truth) reasoning]
{gt_text}

Rate how substantively consistent the model's reasoning is with the reference, on this 1-5 scale:
5 = same analysis -- same region of interest, same attributed effect of the event, same expected behaviour
4 = mostly consistent -- same core analysis, minor difference in emphasis or one secondary detail
3 = partially consistent -- overlaps on the broad situation but differs on a substantive point \
(a different expected effect, or a clearly different region)
2 = mostly inconsistent -- different analysis, only superficial overlap (shared domain vocabulary)
1 = contradictory -- asserts an effect or a region opposite to the reference

Respond with ONLY a JSON object of the form {{"score": <int 1-5>, "reason": "<one short sentence>"}}"""


MERGED_JUDGE_SYSTEM = (
    "You are a careful evaluator comparing two passages that each analyse the same time-series "
    "forecasting task AND state a concluding forecast for it. Judge the passage as a whole: "
    "whether it identifies the same region of interest, attributes the same effect to the stated "
    "event, and concludes with the same shape narrative and magnitude for the future window. "
    "Explicitly IGNORE writing style, ordering, verbosity, and exact clock times / index numbers."
)

MERGED_JUDGE_USER_TEMPLATE = """Compare the two passages below as a whole (reasoning together \
with its concluding forecast statement).

[Model-generated passage]
{gen_text}

[Reference (ground-truth) passage]
{gt_text}

Rate how substantively consistent the model's passage is with the reference, on this 1-5 scale:
5 = same analysis and same conclusion -- same region of interest, same attributed effect, same \
concluding shape and magnitude
4 = mostly consistent -- same core analysis and conclusion, minor difference in emphasis or one \
secondary detail
3 = partially consistent -- overlaps on the broad situation but differs on a substantive point \
(a different expected effect, a clearly different region, or a different concluding shape)
2 = mostly inconsistent -- different analysis and/or conclusion, only superficial overlap \
(shared domain vocabulary)
1 = contradictory -- asserts an effect, a region, or a concluding direction opposite to the reference

Respond with ONLY a JSON object of the form {{"score": <int 1-5>, "reason": "<one short sentence>"}}"""


def split_generation(gen_text):
    """Whole generation -> (reasoning, conclusion). Without a </think> marker the whole text counts as
    the conclusion and the reasoning is empty."""
    g = gen_text or ""
    if "</think>" in g:
        a, b = g.split("</think>", 1)
        return a.strip(), b.strip()
    return "", g.strip()


def build_messages(kind, gen_text, gt_text):
    if kind == "conclusion":
        user = SHAPE_JUDGE_USER_TEMPLATE.format(gen_text=gen_text, gt_conclusion=gt_text)
        return [{"role": "system", "content": SHAPE_JUDGE_SYSTEM}, {"role": "user", "content": user}]
    if kind == "merged":
        user = MERGED_JUDGE_USER_TEMPLATE.format(gen_text=gen_text, gt_text=gt_text)
        return [{"role": "system", "content": MERGED_JUDGE_SYSTEM}, {"role": "user", "content": user}]
    user = REASONING_JUDGE_USER_TEMPLATE.format(gen_text=gen_text, gt_text=gt_text)
    return [{"role": "system", "content": REASONING_JUDGE_SYSTEM}, {"role": "user", "content": user}]


def derangement_partners(n, seed=0):
    """-> ``{i: j}``: a **random derangement** of the whole pool (Sattolo's single-cycle
    permutation): no fixed point, and it **is a permutation**.

    This is the mismatched-partner pairing used on both tasks. Marginally it matches `random_partners`
    (P(partner=j)=1/(n-1)), but every reference is used exactly once, and the multiset of the swapped
    field is conserved, so any metric that depends only on that field is unchanged (an implementation
    self-check).

    Only applicable to a **full-set** evaluation (`sel` = all n rows). On a subset a derangement keeps
    the partners inside the subset; for that case pass `--pairing resample` explicitly.
    Sattolo's algorithm produces single-cycle permutations, a subset of all derangements; the marginal
    distribution of the partner is still uniform.
    """
    order = list(range(n))
    rng = random.Random(seed)
    for i in range(n - 1, 0, -1):          # j strictly below i => a single cycle, no fixed point
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    pairs = {order[i]: order[(i + 1) % n] for i in range(n)}
    assert all(u != v for u, v in pairs.items()), "self-pairing found; the mismatched pairing is invalid"
    assert sorted(pairs.values()) == list(range(n)), "not a permutation"
    return pairs


def random_partners(n_pool, sel, seed=0):
    """For every evaluated sample, draw one partner uniformly at random from **all n_pool rows**
    (excluding itself).

    **For subset evaluations only** (partners come from the whole pool, not from the subset). For a
    full run use `derangement_partners` -- its docstring compares the two. Note that this function's
    mapping **depends on `sel` itself** (the RNG draws one partner per entry, in `sel` order), so
    changing `--n` changes every partner and therefore requires changing `--out` too.
    """
    rng = random.Random(seed)
    out = {}
    for i in sel:
        j = rng.randrange(n_pool - 1)
        out[i] = j if j < i else j + 1
    return out


def build_pairing(n_pool, sel, pairing="derangement", shift=1, seed=0):
    """Pairing dispatch shared by the mismatched-partner scripts (split text judge / plot-vs-text
    judge).

    - ``derangement``: a random derangement of the whole pool (Sattolo permutation), see
      `derangement_partners`. Accepts full runs only (``len(sel) == n_pool``).
    - ``random``: resample independently per row from the whole pool; for subset evaluations, see
      `random_partners`.
    - ``shift``: row i is paired with row (i+shift)%n. This is not a random pairing: the jsonl rows
      are laid out in contiguous per-dataset blocks, so most partners come from the same dataset.

    For the same seed the scripts obtain the same pairing, so their tables can be compared row by row.
    """
    if pairing == "derangement":
        if len(sel) != n_pool:
            raise SystemExit(
                f"--pairing derangement only applies to a full run ({n_pool} rows); got {len(sel)}. "
                "For a subset use --pairing random (resampled independently per row, partners still "
                "drawn from the whole pool)")
        return derangement_partners(n_pool, seed=seed + 1)
    if pairing == "random":
        return random_partners(n_pool, sel, seed=seed + 1)
    if pairing == "shift":
        if n_pool <= shift:
            raise SystemExit("the number of samples must exceed shift, otherwise a row pairs with itself")
        return {i: (i + shift) % n_pool for i in sel}
    raise SystemExit(f"unknown pairing={pairing!r}")


def load_rows(pred_dir):
    """The jsonl rows plus the dataset_name read from the npz in the same directory (the jsonl itself
    does not carry that column; its row order matches the npz)."""
    rows = []
    with open(os.path.join(pred_dir, "forecast_preds.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    import numpy as np
    npz = np.load(os.path.join(pred_dir, "forecast_preds.npz"), allow_pickle=True)
    names = [str(x) for x in npz["dataset_names"]]
    if len(names) != len(rows):
        raise SystemExit(f"row count mismatch: jsonl {len(rows)} vs npz {len(names)}")
    for r, nm in zip(rows, names):
        r["dataset_name"] = nm
    return rows


def run_batch(tasks, client, model, checkpoint_path, max_workers=3):
    """tasks: [{"key","kind","gen","gt", ...}]; resumable, keyed by "key"."""
    done = {}
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    done[r["key"]] = r
    todo = [t for t in tasks if t["key"] not in done]
    print(f"  {len(done)} already scored, {len(todo)} left to run")
    if not todo:
        return done

    def _one(t):
        msgs = build_messages(t["kind"], t["gen"], t["gt"])
        try:
            resp = _with_retry(
                lambda: client.chat.completions.create(model=model, messages=msgs),
                max_attempts=3, base_delay=1.0, retry_exceptions=_retry_exceptions())
            raw = resp.choices[0].message.content
            score, reason = parse_judge_response(raw)
            return {"key": t["key"], "id": t["id"], "kind": t["kind"], "cond": t["cond"],
                    "domain": t["domain"], "score": score, "reason": reason}
        except Exception as e:  # one failed call must not take down the whole batch
            return {"key": t["key"], "id": t["id"], "kind": t["kind"], "cond": t["cond"],
                    "domain": t["domain"], "score": None, "error": f"{type(e).__name__}: {e}"}

    f = open(checkpoint_path, "a", encoding="utf-8") if checkpoint_path else None
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
            for k, out in enumerate(ex.map(_one, todo), 1):
                done[out["key"]] = out
                if f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                    f.flush()
                if k % 20 == 0:
                    print(f"    {k}/{len(todo)}")
    finally:
        if f:
            f.close()
    return done


def agg_by_domain(results, kind, cond):
    """{domain: (mean, n)}, plus an "ALL" entry."""
    buckets = {}
    for r in results.values():
        if r["kind"] != kind or r["cond"] != cond or r.get("score") is None:
            continue
        buckets.setdefault(r["domain"], []).append(r["score"])
        buckets.setdefault("ALL", []).append(r["score"])
    return {d: (sum(v) / len(v), len(v)) for d, v in buckets.items()}


def consistency_by_domain(results, cond):
    """Consistency = the **per-sample mean** of conclusion and reasoning, aggregated afterwards.

    Averaging per sample rather than "averaging the two aggregate means" is what keeps the denominator
    self-consistent when one judge occasionally fails to return a score.
    """
    per = {}
    for r in results.values():
        if r["cond"] != cond or r.get("score") is None or r["kind"] not in ("conclusion", "reasoning"):
            continue
        per.setdefault((r["id"], r["domain"]), {})[r["kind"]] = r["score"]
    buckets = {}
    for (_sid, dom), d in per.items():
        if len(d) != 2:
            continue
        m = (d["conclusion"] + d["reasoning"]) / 2
        buckets.setdefault(dom, []).append(m)
        buckets.setdefault("ALL", []).append(m)
    return {d: (sum(v) / len(v), len(v)) for d, v in buckets.items()}


def agg(results, kind, cond):
    xs = [r["score"] for r in results.values()
          if r["kind"] == kind and r["cond"] == cond and r.get("score") is not None]
    n_fail = sum(1 for r in results.values()
                 if r["kind"] == kind and r["cond"] == cond and r.get("score") is None)
    return (sum(xs) / len(xs) if xs else float("nan")), len(xs), n_fail


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max_workers", type=int, default=3)
    ap.add_argument("--pairing", choices=["derangement", "resample"], default="derangement",
                    help="derangement=random derangement of the whole pool (default, for full runs); "
                         "resample=resample independently per row (for subsets; partners are still "
                         "drawn from the whole pool)")
    ap.add_argument("--merged", action="store_true",
                    help="additionally run the 'merged evaluation': the model's whole "
                         "reasoning+conclusion against the ground truth's whole "
                         "gt_reasoning+gt_conclusion")
    ap.add_argument("--also_shift", action="store_true",
                    help="additionally run a shift-1 pairing (row i paired with row i+1)")
    args = ap.parse_args(argv)

    rows = load_rows(args.pred_dir)
    n_pool = len(rows)
    parts = [split_generation(r.get("gen_text")) for r in rows]

    rng = random.Random(args.seed)
    sel = sorted(rng.sample(range(n_pool), min(args.n, n_pool)))
    if args.pairing == "derangement":
        if len(sel) != n_pool:
            raise SystemExit(
                f"--pairing derangement only applies to a full run (--n {n_pool}); only {len(sel)} "
                "rows are being evaluated. For a subset pass --pairing resample explicitly (partners "
                "are still drawn from the whole pool)")
        partner = derangement_partners(n_pool, seed=args.seed + 1)
    else:
        partner = random_partners(n_pool, sel, seed=args.seed + 1)

    def dsname(i):
        return str(rows[i].get("dataset_name", ""))

    same_ds = sum(1 for i in sel if dsname(i) == dsname(partner[i]))
    print(f"sampled {len(sel)}/{n_pool} rows (seed={args.seed}); pairing={args.pairing}, "
          f"same-dataset rate {100 * same_ds / len(sel):.1f}%")

    tasks = []
    for i in sel:
        rea, con = parts[i]
        j = partner[i]
        rea_j, con_j = parts[j]
        rid = str(rows[i].get("id"))
        dom = domain_of(str(rows[i].get("dataset_name", ""))) or "Unknown"
        gt_con, gt_rea = rows[i].get("gt_conclusion") or "", rows[i].get("gt_reasoning") or ""
        gt_con_j = rows[j].get("gt_conclusion") or ""
        gt_rea_j = rows[j].get("gt_reasoning") or ""
        tasks += [
            {"key": f"conclusion|real|{i}", "id": rid, "kind": "conclusion", "cond": "real",
             "domain": dom, "gen": con, "gt": gt_con},
            {"key": f"conclusion|shuf|{i}", "id": rid, "kind": "conclusion", "cond": "shuf",
             "domain": dom, "gen": con, "gt": gt_con_j},
            {"key": f"reasoning|real|{i}", "id": rid, "kind": "reasoning", "cond": "real",
             "domain": dom, "gen": rea, "gt": gt_rea},
            {"key": f"reasoning|shuf|{i}", "id": rid, "kind": "reasoning", "cond": "shuf",
             "domain": dom, "gen": rea, "gt": gt_rea_j},
        ]
        if args.merged:
            gen_m = (rea + "\n\n" + con).strip()
            gt_m = (gt_rea + "\n\n" + gt_con).strip()
            gt_m_j = ((rows[j].get("gt_reasoning") or "") + "\n\n" + gt_con_j).strip()
            tasks += [
                {"key": f"merged|real|{i}", "id": rid, "kind": "merged", "cond": "real",
                 "domain": dom, "gen": gen_m, "gt": gt_m},
                {"key": f"merged|shuf|{i}", "id": rid, "kind": "merged", "cond": "shuf",
                 "domain": dom, "gen": gen_m, "gt": gt_m_j},
            ]
        if args.also_shift:
            k = (i + 1) % n_pool
            tasks += [
                {"key": f"conclusion|shift|{i}", "id": rid, "kind": "conclusion", "cond": "shift",
                 "domain": dom, "gen": con, "gt": rows[k].get("gt_conclusion") or ""},
                {"key": f"reasoning|shift|{i}", "id": rid, "kind": "reasoning", "cond": "shift",
                 "domain": dom, "gen": rea, "gt": rows[k].get("gt_reasoning") or ""},
            ]
            if args.merged:
                gt_m_k = ((rows[k].get("gt_reasoning") or "") + "\n\n"
                          + (rows[k].get("gt_conclusion") or "")).strip()
                tasks.append({"key": f"merged|shift|{i}", "id": rid, "kind": "merged",
                              "cond": "shift", "domain": dom,
                              "gen": (rea + "\n\n" + con).strip(), "gt": gt_m_k})
        _ = (rea_j, con_j)  # the partner's generated text is unused here (only the reference is swapped)

    os.makedirs(args.out, exist_ok=True)
    client = build_openai_client()
    print(f"{len(tasks)} calls in total ({len(sel)} rows x 2 judges x 2 pairings), model={args.model}")
    results = run_batch(tasks, client, args.model,
                        os.path.join(args.out, "judge_raw.jsonl"),
                        max_workers=args.max_workers)

    lines = [f"# Forecasting side: split evaluation + mismatched pairing (n={len(sel)}, seed={args.seed})\n",
             f"Source: `{args.pred_dir}`; judge={args.model}; "
             f"same-dataset rate of the partners {100 * same_ds / len(sel):.1f}%\n",
             "| judge | matched | mismatched | Δ | scoring failures |", "|---|---|---|---|---|"]
    kinds = [("conclusion", "conclusion ↔ gt_conclusion (shape judge)"),
             ("reasoning", "reasoning ↔ gt_reasoning")]
    if args.merged:
        kinds.append(("merged", "merged: reasoning+conclusion ↔ gt_reasoning+gt_conclusion"))
    for kind, label in kinds:
        mr, nr, fr = agg(results, kind, "real")
        ms, ns, fs = agg(results, kind, "shuf")
        lines.append(f"| {label} | **{mr:.3f}** (n={nr}) | **{ms:.3f}** (n={ns}) | "
                     f"**{ms - mr:+.3f}** | {fr + fs} |")
    if args.also_shift:
        lines += ["", "Shift-1 pairing for reference:", "",
                  "| judge | shift-1 mismatch | mismatched | difference |", "|---|---|---|---|"]
        for kind, _lab in kinds:
            mh, nh, _ = agg(results, kind, "shift")
            ms, _, _ = agg(results, kind, "shuf")
            lines.append(f"| {kind} | {mh:.3f} (n={nh}) | {ms:.3f} | {ms - mh:+.3f} |")
    # Consistency = the per-sample mean of conclusion and reasoning
    cr, cs = consistency_by_domain(results, "real"), consistency_by_domain(results, "shuf")
    if cr.get("ALL"):
        lines.append(f"| **consistency (per-sample mean of conclusion and reasoning)** | "
                     f"**{cr['ALL'][0]:.3f}** (n={cr['ALL'][1]}) | **{cs['ALL'][0]:.3f}** "
                     f"(n={cs['ALL'][1]}) | **{cs['ALL'][0] - cr['ALL'][0]:+.3f}** | — |")

    doms = ["ALL", "Solar", "Load", "Traffic", "Finance", "Climate"]
    lines += ["", "Per domain (matched / mismatched / Δ):", "",
              "| metric | " + " | ".join(doms) + " |", "|---" * (len(doms) + 1) + "|"]
    rows_spec = [("conclusion", "conclusion"), ("reasoning", "reasoning")]
    for kind, label in rows_spec:
        R, S = agg_by_domain(results, kind, "real"), agg_by_domain(results, kind, "shuf")
        cells = []
        for d in doms:
            if d in R and d in S:
                cells.append(f"{R[d][0]:.3f} / {S[d][0]:.3f} / {S[d][0] - R[d][0]:+.3f}")
            else:
                cells.append("—")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    cells = []
    for d in doms:
        if d in cr and d in cs:
            cells.append(f"**{cr[d][0]:.3f}** / **{cs[d][0]:.3f}** / **{cs[d][0] - cr[d][0]:+.3f}**")
        else:
            cells.append("—")
    lines.append("| **consistency (mean)** | " + " | ".join(cells) + " |")

    # Per-sample detail, so the qualitative scores can later be re-aggregated over a filtered subset
    # of the rows without re-running the judge.
    import csv as _csv
    per = {}
    for r in results.values():
        if r.get("score") is None:
            continue
        per.setdefault((r["id"], r["domain"]), {})[f"{r['kind']}_{r['cond']}"] = r["score"]
    ps = os.path.join(args.out, "per_sample.csv")
    cols = ["id", "domain", "conclusion_real", "conclusion_shuf", "reasoning_real",
            "reasoning_shuf", "consistency_real", "consistency_shuf"]
    with open(ps, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for (sid, dom), d in sorted(per.items()):
            row = {"id": sid, "domain": dom}
            row.update({k: d.get(k, "") for k in cols[2:6]})
            for cond in ("real", "shuf"):
                a, b = d.get(f"conclusion_{cond}"), d.get(f"reasoning_{cond}")
                row[f"consistency_{cond}"] = "" if a is None or b is None else (a + b) / 2
            w.writerow(row)

    text = "\n".join(lines) + "\n"
    print("\n" + text)
    with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as f:
        f.write(text)
    print(f"artefacts written to {args.out}/")


if __name__ == "__main__":
    main()
