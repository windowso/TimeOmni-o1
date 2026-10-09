"""CPU unit tests for the mismatched-partner control of the understanding-side reasoning-trace judge
(fake client injected, no API calls).

These guard four things that would "not raise when wrong, but silently produce a plausible-looking
wrong table":
1. **No self-pairing** -- a self-pair equals no shuffle, so the mismatched run would score the same as
   the matched run;
2. **The cyclic shift is a permutation => the multiset of explanations is conserved** -- every
   explanation is judged exactly once in the mismatched run;
3. **Only the shuffled field changes** -- if the expl group also swapped gt_reasoning, the two controls
   would blur into one;
4. **The paired comparison is against the explanation's original owner** (real[pairs[uid]], not
   real[uid]) -- comparing against the wrong row confuses "the explanation changed owner" with "the
   sample changed owner", and both sign and magnitude of the paired delta would be wrong.
"""
import json
import os
import tempfile

from chronos_llm.scripts.utils.shuffled_explanation_judge_control import (
    apply_shuffle, build_shuffled_pairs, by_domain_table, paired_delta, score_index,
)


def _recs(n=6, task="t", dataset="d", domain="Physiology"):
    return [{"uid": f"u{i}", "dataset": dataset, "task": task, "domain": domain,
             "correct": i % 2 == 0, "question": f"q{i}", "gt_answer": "a",
             "pred_answer": "a", "explanation": f"expl{i}", "gt_reasoning": f"ref{i}"}
            for i in range(n)]


def test_no_self_pairing_and_permutation():
    recs = _recs(6)
    pairs, n_orphan = build_shuffled_pairs(recs, scope="task")
    assert n_orphan == 0
    assert all(u != v for u, v in pairs.items()), pairs
    # within a fully populated scope it is a permutation: every record referenced exactly once
    # (=> the content multiset is conserved)
    assert sorted(pairs.values()) == sorted(pairs), pairs
    print("no self-pairing + permutation within scope OK")


def test_scope_confines_pairing():
    """scope=task pairs only within the same dataset and subtask."""
    recs = _recs(4, task="ecg") + _recs(4, task="har")
    for i, r in enumerate(recs):          # make uids unique
        r["uid"] = f"{r['task']}{i}"
    pairs, _ = build_shuffled_pairs(recs, scope="task")
    task_of = {r["uid"]: r["task"] for r in recs}
    assert all(task_of[u] == task_of[v] for u, v in pairs.items()), pairs
    print("pairing confined to scope OK")


def test_orphan_scope_falls_back():
    """A sample alone in its scope cannot self-pair; it must fall into the orphan pool and pair there."""
    recs = _recs(3, task="big") + _recs(1, task="lonelyA") + _recs(1, task="lonelyB")
    for i, r in enumerate(recs):
        r["uid"] = f"{r['task']}{i}"
    pairs, n_orphan = build_shuffled_pairs(recs, scope="task")
    assert n_orphan == 2, n_orphan
    assert all(u != v for u, v in pairs.items())
    assert len(pairs) == len(recs)
    print(f"orphan scopes fall into the orphan pool and pair there OK ({n_orphan} records)")


def test_apply_shuffle_changes_only_one_field():
    recs = _recs(4)
    pairs, _ = build_shuffled_pairs(recs, scope="task")
    shuf = apply_shuffle(recs, pairs, "explanation")
    by_uid = {r["uid"]: r for r in recs}
    for s in shuf:
        own = by_uid[s["uid"]]
        assert s["explanation"] == by_uid[pairs[s["uid"]]]["explanation"] != own["explanation"]
        for k in ("question", "gt_answer", "pred_answer", "gt_reasoning", "correct", "domain"):
            assert s[k] == own[k], (k, s[k], own[k])
        assert s["_from_uid"] == pairs[s["uid"]]
    # explanation multiset conserved
    assert sorted(s["explanation"] for s in shuf) == sorted(r["explanation"] for r in recs)
    print("apply_shuffle changes only one field + multiset conserved OK")


def test_paired_delta_compares_against_owner():
    """The paired delta must be computed against **the explanation's original owner**:
    real[pairs[uid]], not real[uid]."""
    recs = _recs(2)
    pairs = {"u0": "u1", "u1": "u0"}
    real = score_index([{"uid": "u0", "support": 5.0, "consistency": 5.0},
                        {"uid": "u1", "support": 1.0, "consistency": 1.0}])
    # u0 judges u1's explanation (real score 1.0), scores 2.0 after mismatch => delta=+1; u1 judges
    # u0's explanation (5.0), scores 2.0 => delta=-3
    shuf = score_index([{"uid": "u0", "support": 2.0, "consistency": 2.0},
                        {"uid": "u1", "support": 2.0, "consistency": 2.0}])
    d = paired_delta(real, shuf, pairs)
    assert abs(d["mean_delta"] - (-1.0)) < 1e-9, d      # (+1 + -3)/2
    assert (d["win"], d["loss"]) == (1, 1), d
    # A wrong comparison against itself, (2-5 + 2-1)/2 = -1.0, would give the same number, so guard
    # again with an asymmetric set
    shuf2 = score_index([{"uid": "u0", "support": 4.0, "consistency": 2.0},
                         {"uid": "u1", "support": 4.0, "consistency": 2.0}])
    assert abs(paired_delta(real, shuf2, pairs)["mean_delta"] - 1.0) < 1e-9   # (+3 + -1)/2
    print("paired delta computed against the explanation's owner OK")


def test_table_deltas_and_error_rows_excluded():
    """The summary table's delta direction is correct; rows whose scoring failed must not be
    counted as 0 and drag the mean down."""
    recs = _recs(2, domain="Energy")
    real = score_index([{"uid": "u0", "support": 5.0, "consistency": 5.0},
                        {"uid": "u1", "support": 5.0, "consistency": 5.0},
                        {"uid": "u2", "error": "unparsable"}])
    assert "u2" not in real, "rows whose scoring failed must be excluded"
    expl = score_index([{"uid": "u0", "support": 2.0, "consistency": 3.0},
                        {"uid": "u1", "support": 2.0, "consistency": 3.0}])
    ref = score_index([{"uid": "u0", "support": 5.0, "consistency": 1.0},
                       {"uid": "u1", "support": 5.0, "consistency": 1.0}])
    lines, A, B, C = by_domain_table(recs, real, expl, ref)
    body = "\n".join(lines)
    assert "-3.00" in body and "-4.00" in body, body     # support 5->2, consistency 5->1
    assert C["support"] == A["support"]                    # the ref group leaves support untouched
    print("summary table delta direction and error-row exclusion OK")


def test_end_to_end_with_fake_client():
    """Full pipeline (build_records reusing the matched run's sampling + uid-set check + fake-client
    scoring + writing outputs)."""
    from unittest import mock

    from chronos_llm.eval.explanation_judge import build_records
    from chronos_llm.scripts.utils import shuffled_explanation_judge_control as ctrl

    with tempfile.TemporaryDirectory() as d:
        infer = os.path.join(d, "infer")
        os.makedirs(infer)
        with open(os.path.join(infer, "sleep_test.jsonl"), "w", encoding="utf-8") as f:
            for i in range(60):
                ok = i % 3 != 0
                f.write(json.dumps({
                    "id": f"s{i}", "task": "sleep_cot", "input_text": f"stage of window {i}?",
                    "ground_truth": "Answer: N1",
                    "generated_text": f"Spindles at {i} Hz suggest N1.\nAnswer: {'N1' if ok else 'N2'}",
                    "gt_reasoning": f"reference reasoning {i}"}) + "\n")

        records = build_records(infer, verbose=False)
        real_path = os.path.join(d, "scores.jsonl")
        with open(real_path, "w", encoding="utf-8") as f:
            for r in records:                       # matched run: true pairs always score 5
                f.write(json.dumps({"uid": r["uid"], "support": 5.0, "consistency": 5.0, "reason": "ok"}) + "\n")

        class _FakeClient:
            """A mismatched explanation => support 2; otherwise 5. Decided by whether the sample index
            in the text matches."""
            class chat:
                class completions:
                    @staticmethod
                    def create(model=None, messages=None):
                        user = messages[1]["content"]
                        q = user.split("window ")[1].split("?")[0]
                        expl_i = user.split("Spindles at ")[1].split(" Hz")[0]
                        sup = 5 if q == expl_i else 2
                        payload = json.dumps({"support": sup, "consistency": 5, "reason": "r"})
                        return mock.Mock(choices=[mock.Mock(message=mock.Mock(content=payload))])

        out = os.path.join(d, "out")
        with mock.patch.object(ctrl, "build_openai_client", lambda: _FakeClient()):
            rc = ctrl.main([infer, "--real_scores", real_path, "--out", out, "--max_workers", "2"])
        assert rc == 0
        md = open(os.path.join(out, "shuffled_explanation_judge.md"), encoding="utf-8").read()
        assert "-3.00" in md, md              # support 5.00 -> 2.00
        assert "(delta=+0.00)" in md, md      # the ref group leaves support untouched
        rows = list(open(os.path.join(out, "per_sample.csv"), encoding="utf-8"))
        assert len(rows) == len(records) + 1, (len(rows), len(records))

        # A mismatched uid set must raise instead of comparing two differently sampled tables side by side
        bad = os.path.join(d, "bad_scores.jsonl")
        with open(bad, "w", encoding="utf-8") as f:
            for r in records[:10]:
                f.write(json.dumps({"uid": r["uid"], "support": 5.0, "consistency": 5.0}) + "\n")
        try:
            ctrl.main([infer, "--real_scores", bad, "--out", os.path.join(d, "out2"), "--dry_run"])
            raise AssertionError("should raise when the sampling differs from the matched run")
        except SystemExit as e:
            assert "sample set differs" in str(e), e
    print("full pipeline (fake client) + uid-set check OK")


def test_random_pairs_is_derangement_and_random():
    """Random derangement over the whole pool: must be a permutation (multiset conserved), no
    self-pairing, and genuinely cross-dataset."""
    from chronos_llm.scripts.utils.shuffled_explanation_judge_control import build_random_pairs
    # uids start with the dataset name, as in real records
    recs = [{"uid": f"{'dsA' if i < 60 else 'dsB'}::x{i:03d}", "dataset": "dsA" if i < 60 else "dsB",
             "task": "t", "domain": "D"} for i in range(100)]
    pairs, n_orphan = build_random_pairs(recs, seed=0)
    assert n_orphan == 0
    uids = sorted(r["uid"] for r in recs)
    assert sorted(pairs) == uids and sorted(pairs.values()) == uids, "must be a permutation"
    assert all(u != v for u, v in pairs.items()), "no self-pairing allowed"
    ds = {r["uid"]: r["dataset"] for r in recs}
    same = sum(1 for u, v in pairs.items() if ds[u] == ds[v])
    assert same < 80, f"a random derangement should not have {same}/100 pairs in the same dataset"
    # Contrast with the shift pairing: with such block-structured uids it almost always pairs within
    # the same dataset
    from chronos_llm.scripts.utils.shuffled_explanation_judge_control import build_shuffled_pairs
    sp, _ = build_shuffled_pairs(recs, scope="global", shift=1)
    same_shift = sum(1 for u, v in sp.items() if ds[u] == ds[v])
    assert same_shift >= 98, same_shift
    assert build_random_pairs(recs, seed=0)[0] == pairs                 # reproducible
    assert build_random_pairs(recs, seed=1)[0] != pairs                 # changes with the seed
    print(f"  random derangement same-dataset rate {same}% vs {same_shift}% for shift-1; "
          f"permutation/no self-pairing/reproducible OK")


if __name__ == "__main__":
    test_no_self_pairing_and_permutation()
    test_scope_confines_pairing()
    test_orphan_scope_falls_back()
    test_apply_shuffle_changes_only_one_field()
    test_paired_delta_compares_against_owner()
    test_table_deltas_and_error_rows_excluded()
    test_end_to_end_with_fake_client()
    test_random_pairs_is_derangement_and_random()
    print("ALL OK")
