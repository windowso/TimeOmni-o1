r"""Understanding side: mismatched-partner control for the reasoning-trace judge
(`eval/explanation_judge.py`).

Two mismatched runs, each breaking exactly one pairing relation (prompt unchanged, same `judge_one`):

- **Group expl (control for supportiveness)**: replace `explanation` with the explanation of another
  sample; question / model answer / reference reasoning stay the sample's own. This
  breaks the "explanation <-> the model's own answer" pair on which support is defined.
- **Group ref (control for consistency)**: replace `gt_reasoning` (the annotated reference trace) with
  another sample's; everything else unchanged. This breaks the "explanation <-> reference reasoning"
  pair.

Implementation self-check: group ref leaves explanation and answer untouched, so support should
match the matched run.

Pairing: `--pairing random` (default) draws a uniform derangement over the whole pool;
`--pairing shift` cyclically shifts within `--scope` (default: same dataset and sub-task).

Sampling is not re-implemented: `explanation_judge.build_records()` is called with the same
parameters and seed, so the uids match the matched run; at start-up the script checks the uid set
against the matched run's `scores.jsonl` and fails on mismatch.

    python chronos_llm/scripts/utils/shuffled_explanation_judge_control.py \
        outputs/eval/<run>/infer \
        --real_scores outputs/eval/<run>/explanation_judge/scores.jsonl \
        --out outputs/eval/<run>/explanation_judge_shuffled \
        --pairing random --skip_empty_explanation

Note: this script needs outbound network access to the judge API.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, REPO)

from chronos_llm.eval.explanation_judge import (  # noqa: E402
    DEFAULT_JUDGE_MODEL, build_openai_client, build_records, run_judge_batch,
)

SCOPE_KEYS = {"task": ("dataset", "task"), "dataset": ("dataset",), "domain": ("domain",),
              "global": ()}


def build_random_pairs(records, seed=0):
    """-> ``{uid: partner_uid}``: a **uniformly random derangement over the whole pool**, ignoring scope.

    With `--pairing shift`, records are sorted by uid and cyclically shifted; since uids start with the
    dataset name, neighbouring uids almost always belong to the same dataset. This function instead
    draws partners from the whole pool.

    Uses **Sattolo's algorithm** to generate a single-cycle permutation: guarantees **no fixed points**
    (no self-pairing) while still being a **permutation** => every explanation is judged exactly once
    in group expl.
    """
    uids = sorted(r["uid"] for r in records)
    n = len(uids)
    if n < 2:
        raise ValueError("fewer than 2 samples, cannot build shuffled pairs")
    order = list(uids)
    rng = random.Random(seed)
    for i in range(n - 1, 0, -1):          # Sattolo: j strictly < i => single cycle, no fixed points
        j = rng.randrange(i)
        order[i], order[j] = order[j], order[i]
    pairs = {order[i]: order[(i + 1) % n] for i in range(n)}
    assert all(u != v for u, v in pairs.items()), "self-pairing found, mismatched pairing invalid"
    assert sorted(pairs.values()) == uids, "not a permutation, explanation multiset not conserved"
    return pairs, 0


def build_shuffled_pairs(records, scope="task", shift=1):
    """-> ``{uid: partner_uid}``: sort uids within each scope and cyclically shift by ``shift``.

    A scope containing **only 1** sample cannot be paired within itself (it would self-pair); such
    samples are collected into an orphan pool that is cyclically shifted on
    its own; only when the pool itself has a single sample do we fall back to any other global sample.
    The returned mapping is guaranteed **self-pairing free**, and within every full scope it is a
    **permutation** (=> every explanation is judged exactly once in group expl).
    """
    keys = SCOPE_KEYS[scope]
    buckets = defaultdict(list)
    for r in records:
        buckets[tuple(r[k] for k in keys)].append(r["uid"])

    pairs, orphans = {}, []
    for key in sorted(buckets):
        uids = sorted(buckets[key])
        if len(uids) < 2:
            orphans.extend(uids)
            continue
        s = shift % len(uids) or 1
        for i, u in enumerate(uids):
            pairs[u] = uids[(i + s) % len(uids)]

    if orphans:
        orphans = sorted(orphans)
        if len(orphans) >= 2:
            for i, u in enumerate(orphans):
                pairs[u] = orphans[(i + 1) % len(orphans)]
        else:
            u = orphans[0]
            others = sorted(r["uid"] for r in records if r["uid"] != u)
            if not others:
                raise ValueError("fewer than 2 samples, cannot build shuffled pairs")
            pairs[u] = others[0]

    assert all(u != v for u, v in pairs.items()), "self-pairing found, mismatched pairing invalid"
    assert len(pairs) == len(records), (len(pairs), len(records))
    return pairs, len(orphans)


def apply_shuffle(records, pairs, field):
    """Replace ``field`` of every record with the same field of its paired sample; keep everything else.

    Additionally writes ``_from_uid`` recording the true owner of the swapped content, which the
    per-sample table uses for paired comparison.
    """
    by_uid = {r["uid"]: r for r in records}
    out = []
    for r in records:
        partner = by_uid[pairs[r["uid"]]]
        rec = dict(r)
        rec[field] = partner[field]
        rec["_from_uid"] = partner["uid"]
        out.append(rec)
    return out


def score_index(scored):
    """Score list -> ``{uid: score_dict}``, skipping error rows (not silently counted as 0)."""
    return {s["uid"]: s for s in scored if not s.get("error") and "support" in s}


def agg(scores):
    """A group of scores -> mean support and consistency."""
    if not scores:
        return {"n": 0, "support": float("nan"), "consistency": float("nan")}
    n = len(scores)
    return {
        "n": n,
        "support": sum(s["support"] for s in scores) / n,
        "consistency": sum(s["consistency"] for s in scores) / n,
    }


def paired_delta(real_idx, shuf_idx, pairs, field_owner_is_partner=True):
    """**Paired** support difference of the same explanation under matched vs mismatched pairing.

    In group expl the record with uid=j is judged on the explanation of uid=i (i=pairs[j] is the
    owner), so it must be compared against ``real[i]``, not ``real[j]`` -- otherwise "the explanation
    changed owner" gets confused with "the sample changed".
    """
    deltas, win, tie, loss = [], 0, 0, 0
    for uid, s in shuf_idx.items():
        owner = pairs[uid] if field_owner_is_partner else uid
        r = real_idx.get(owner)
        if r is None:
            continue
        d = s["support"] - r["support"]
        deltas.append(d)
        win += d > 0
        tie += d == 0
        loss += d < 0
    mean = sum(deltas) / len(deltas) if deltas else float("nan")
    return {"n": len(deltas), "mean_delta": mean, "win": win, "tie": tie, "loss": loss}


def by_domain_table(records, real_idx, expl_idx, ref_idx):
    """-> markdown rows: one row per discipline, real / both shuffles side by side."""
    dom_of = {r["uid"]: r["domain"] or "Unknown" for r in records}
    groups = defaultdict(lambda: defaultdict(list))
    for name, idx in (("real", real_idx), ("expl", expl_idx), ("ref", ref_idx)):
        for uid, s in idx.items():
            groups[dom_of.get(uid, "Unknown")][name].append(s)

    lines = ["| discipline | n | support real | support shuf-expl | **delta (lower=better)** | consistency real "
             "| consistency shuf-ref | **delta (lower=better)** |", "|---|---|---|---|---|---|---|---|"]
    for dom in sorted(groups):
        a, b, c = (agg(groups[dom][k]) for k in ("real", "expl", "ref"))
        lines.append(
            f"| {dom} | {a['n']} | {a['support']:.2f} | {b['support']:.2f} "
            f"| **{b['support'] - a['support']:+.2f}** | {a['consistency']:.2f} "
            f"| {c['consistency']:.2f} | **{c['consistency'] - a['consistency']:+.2f}** |")
    A, B, C = (agg([s for g in groups.values() for s in g[k]]) for k in ("real", "expl", "ref"))
    lines.append(
        f"| **All** | **{A['n']}** | **{A['support']:.2f}** | **{B['support']:.2f}** "
        f"| **{B['support'] - A['support']:+.2f}** | **{A['consistency']:.2f}** "
        f"| **{C['consistency']:.2f}** | **{C['consistency'] - A['consistency']:+.2f}** |")
    return lines, A, B, C


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("infer_dir")
    ap.add_argument("--real_scores", required=True,
                    help="scores.jsonl of the matched run (side-by-side comparison + uid-set consistency check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--scope", choices=tuple(SCOPE_KEYS), default="task",
                    help="pairing scope for --pairing shift: task (default, same dataset & sub-task) / dataset / domain / global")
    ap.add_argument("--shift", type=int, default=1)
    ap.add_argument("--pairing", choices=["shift", "random"], default="random",
                    help="shift = cyclic shift within scope (mostly same-dataset partners); random = uniform "
                         "derangement over the whole pool (Sattolo, still a permutation => self-checks stay valid)")
    ap.add_argument("--by", choices=("domain", "dataset"), default="domain")
    ap.add_argument("--per_dataset", type=int, default=100)
    ap.add_argument("--budget_k", type=float, default=2.0)
    ap.add_argument("--floor", type=int, default=40)
    ap.add_argument("--cap", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_empty_explanation", action="store_true")
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL)
    ap.add_argument("--max_workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0,
                    help=">0: judge only the first N records per group (sorted by uid; saves API cost)")
    ap.add_argument("--dry_run", action="store_true", help="only build the pairs, do not call the API")
    args = ap.parse_args(argv)

    records = build_records(
        args.infer_dir, by=args.by, per_dataset=args.per_dataset, budget_k=args.budget_k,
        floor=args.floor, cap=args.cap, seed=args.seed,
        skip_empty_explanation=args.skip_empty_explanation)

    real_scored = [json.loads(l) for l in open(args.real_scores, encoding="utf-8")]
    real_idx = score_index(real_scored)
    uids, real_uids = {r["uid"] for r in records}, {s["uid"] for s in real_scored}
    if uids != real_uids:
        raise SystemExit(
            f"[control] sample set differs from the matched run: this run {len(uids)} / matched run {len(real_uids)}, "
            f"only here {len(uids - real_uids)}, only in matched run {len(real_uids - uids)}. "
            f"Side-by-side comparison requires identical sampling -- align --by/--budget_k/--floor/--cap/--seed/"
            f"--skip_empty_explanation and rerun.")
    print(f"[control] uid set identical to the matched run ({len(uids)} records)")

    if args.pairing == "random":
        pairs, n_orphan = build_random_pairs(records, seed=args.seed + 1)
    else:
        pairs, n_orphan = build_shuffled_pairs(records, scope=args.scope, shift=args.shift)
    dom_of = {r["uid"]: r["domain"] for r in records}
    ds_of = {r["uid"]: r["dataset"] for r in records}
    same_dom = sum(1 for u, v in pairs.items() if dom_of[u] == dom_of[v])
    same_ds = sum(1 for u, v in pairs.items() if ds_of[u] == ds_of[v])
    # The pairing protocol is spelled out in the artifacts, so a reader can tell shift from random pairing.
    pairing_desc = (f"pairing=random (uniform derangement over the pool, Sattolo) seed={args.seed + 1}"
                    if args.pairing == "random"
                    else f"pairing=shift (cyclic shift within scope={args.scope}, shift={args.shift})")
    print(f"[control] {pairing_desc}: {len(pairs)} pairs, no self-pairing, {n_orphan} orphans, "
          f"same dataset {same_ds}/{len(pairs)} ({100 * same_ds / len(pairs):.1f}%), "
          f"same domain {same_dom}/{len(pairs)}")

    shuf_expl = apply_shuffle(records, pairs, "explanation")
    shuf_ref = apply_shuffle(records, pairs, "gt_reasoning")
    if args.limit > 0:
        keep = {r["uid"] for r in sorted(records, key=lambda x: x["uid"])[:args.limit]}
        shuf_expl = [r for r in shuf_expl if r["uid"] in keep]
        shuf_ref = [r for r in shuf_ref if r["uid"] in keep]
        print(f"[control] --limit {args.limit}: judging only {len(shuf_expl)} records per group")
    if args.dry_run:
        for r in shuf_expl[:2]:
            print(f"\n  uid={r['uid']}\n  explanation taken from {r['_from_uid']}\n  "
                  f"first 120 chars of explanation: {r['explanation'][:120]!r}")
        return 0

    os.makedirs(args.out, exist_ok=True)
    client = build_openai_client()
    print(f"\n[control] group expl (support control, swapped explanation) n={len(shuf_expl)}")
    expl_idx = score_index(run_judge_batch(
        shuf_expl, os.path.join(args.out, "scores_shuffle_expl.jsonl"), client,
        model=args.model, max_workers=args.max_workers))
    print(f"\n[control] group ref (consistency control, swapped reference reasoning) n={len(shuf_ref)}")
    ref_idx = score_index(run_judge_batch(
        shuf_ref, os.path.join(args.out, "scores_shuffle_ref.jsonl"), client,
        model=args.model, max_workers=args.max_workers))

    # Compare only on uids scored in all three groups (same denominator, so a failed judge call in one
    # group cannot give the three columns different bases)
    common = set(real_idx) & set(expl_idx) & set(ref_idx)
    real_c = {u: real_idx[u] for u in common}
    expl_c = {u: expl_idx[u] for u in common}
    ref_c = {u: ref_idx[u] for u in common}
    lines, A, B, C = by_domain_table(records, real_c, expl_c, ref_c)

    pd_expl = paired_delta(real_idx, expl_c, pairs)
    supp_ref_gap = C["support"] - A["support"]

    md = "\n".join([
        "# Reasoning-trace judge: mismatched-partner control", "",
        f"infer_dir=`{args.infer_dir}`  judge={args.model}  **{pairing_desc}**  "
        f"same-dataset pairing rate **{100 * same_ds / len(pairs):.1f}%**  "
        f"{len(common)} records comparable across all three groups (sampling identical to the matched run)", "",
        "Each shuffle breaks exactly one pairing relation, prompt unchanged: **group \"shuffled explanation\"** "
        "replaces the explanation with another sample's (breaks explanation <-> the model's own answer, "
        "targeting the definition of support); **group \"shuffled reference\"** replaces the annotated reference "
        "reasoning with another sample's (breaks explanation <-> reference reasoning, targeting the definition "
        "of consistency).", "",
    ] + lines + [
        "", "## Self-check", "",
        f"**Support with a shuffled reference**: group \"shuffled reference\" leaves explanation and answer "
        f"untouched => support should not change. Observed {A['support']:.2f} -> {C['support']:.2f} "
        f"(delta={supp_ref_gap:+.2f}).", "",
        "## Paired comparison (same explanation: matched vs mismatched pairing)", "",
        f"n={pd_expl['n']}, paired support difference **{pd_expl['mean_delta']:+.2f}**, "
        f"mismatched higher {pd_expl['win']} / tie {pd_expl['tie']} / mismatched lower {pd_expl['loss']}.",
        "(The same explanation is judged once under each pairing and compared pairwise.)",
    ])
    out_md = os.path.join(args.out, "shuffled_explanation_judge.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print("\n" + md)

    csv_path = os.path.join(args.out, "per_sample.csv")
    by_uid = {r["uid"]: r for r in records}
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "uid", "domain", "dataset", "task", "correct", "paired_with",
            "real_support", "shuf_expl_support", "shuf_ref_support",
            "real_consistency", "shuf_expl_consistency", "shuf_ref_consistency"])
        w.writeheader()
        for uid in sorted(common):
            r = by_uid[uid]
            w.writerow({
                "uid": uid, "domain": r["domain"] or "Unknown", "dataset": r["dataset"],
                "task": r["task"], "correct": r["correct"], "paired_with": pairs[uid],
                "real_support": real_c[uid]["support"],
                "shuf_expl_support": expl_c[uid]["support"],
                "shuf_ref_support": ref_c[uid]["support"],
                "real_consistency": real_c[uid]["consistency"],
                "shuf_expl_consistency": expl_c[uid]["consistency"],
                "shuf_ref_consistency": ref_c[uid]["consistency"],
            })
    print(f"\n[control] wrote {out_md} and {csv_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
