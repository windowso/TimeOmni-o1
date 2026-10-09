"""Build a shuffled-event test set for the `plain_prompt` column: replace each row's "event
paragraph" with the event paragraph of another row from the same dataset (the background paragraph
and the time/window metadata paragraph stay unchanged and still align with the true values), so that
the only variable is "whether the event content is correct".

plain_prompt is a three-paragraph template (separated by blank lines):
  [0] dataset background (generic description, identical for all rows of the dataset)
  [1] event paragraph (differs per row; carries the actual "what happened today" information; the
      only part this script modifies)
  [2] series metadata (indices and timestamps of the history/future windows; must align with the
      true values and must not be touched)

Replacement strategy: only within split=test rows, grouped by dataset_name; the event paragraphs
within a group are cyclically shifted by 1 (rows are pairwise offset within the group, so no row
gets its own original event back).
Background/metadata paragraphs and every column other than plain_prompt (history_values /
future_values / roi_* / ...) are left untouched; predictions are still scored against the true
values, and the only variable is "whether the event description fed to the LLM is what really
happened for this row".

Usage:
  python chronos_llm/scripts/utils/make_shuffled_event_plain_prompt.py \\
      --in  <corpus_dataset.parquet> \\
      --out <corpus_dataset_eventshuffled.parquet>
"""
import argparse

import pandas as pd


def _split_paragraphs(text):
    return text.split("\n\n")


def shuffle_event_paragraph(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    new_plain = out["plain_prompt"].astype(object).copy()
    src_id_col = pd.Series([None] * len(out), index=out.index, dtype=object)

    test_mask = out["split"] == "test"
    n_touched = 0
    n_paragraph_mismatch = 0
    for ds, group in out[test_mask].groupby("dataset_name"):
        idx = group.index.tolist()
        n = len(idx)
        if n < 2:
            # A single-row group cannot be offset (skip defensively instead of raising).
            continue
        paras_per_row = {}
        ok_idx = []
        for i in idx:
            paras = _split_paragraphs(out.at[i, "plain_prompt"])
            if len(paras) < 3:
                n_paragraph_mismatch += 1
                continue
            paras_per_row[i] = paras
            ok_idx.append(i)
        m = len(ok_idx)
        if m < 2:
            continue
        for pos, i in enumerate(ok_idx):
            donor = ok_idx[(pos + 1) % m]  # cyclic shift by 1; donor != i always holds (m>=2)
            assert donor != i
            paras = list(paras_per_row[i])
            donor_paras = paras_per_row[donor]
            paras[1] = donor_paras[1]  # swap only the event paragraph; background/metadata stay
            new_plain.at[i] = "\n\n".join(paras)
            src_id_col.at[i] = out.at[donor, "id"] if "id" in out.columns else str(donor)
            n_touched += 1

    out["plain_prompt"] = new_plain
    out["event_shuffled_from_id"] = src_id_col  # provenance column, never reaches the model (the dataset ignores unknown columns)
    stats = {
        "n_test_rows": int(test_mask.sum()),
        "n_touched": n_touched,
        "n_paragraph_mismatch": n_paragraph_mismatch,
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="dst", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    if "plain_prompt" not in df.columns:
        raise ValueError("parquet has no plain_prompt column; generate it first with build_plain_prompt.py --apply")

    out, stats = shuffle_event_paragraph(df)

    # Defensive check: the replaced event paragraph of every row really differs from the original
    # (guards against a buggy cyclic-shift implementation that changes nothing).
    test_rows = out[out["split"] == "test"]
    orig = df.set_index(df.index)
    n_actually_changed = 0
    for i in test_rows.index:
        if out.at[i, "event_shuffled_from_id"] is not None:
            old_paras = _split_paragraphs(orig.at[i, "plain_prompt"])
            new_paras = _split_paragraphs(out.at[i, "plain_prompt"])
            if len(old_paras) >= 2 and len(new_paras) >= 2 and old_paras[1] != new_paras[1]:
                n_actually_changed += 1
    n_coincident = stats["n_touched"] - n_actually_changed
    # A handful of rows may receive a donor event paragraph whose text coincides with their own
    # (e.g. the placeholder sentence "no usable news") as long as the donor row index really differs
    # (the shuffle logic itself is fine); only an abnormally large fraction indicates a bug.
    assert n_coincident <= max(5, stats["n_touched"] * 0.02), (
        f"Check failed: claimed to modify {stats['n_touched']} rows, but only {n_actually_changed} rows "
        f"actually changed their event paragraph text ({n_coincident} coincidentally identical, above the "
        f"2% tolerance) -- inspect the shuffle logic first."
    )
    if n_coincident:
        print(f"Note: {n_coincident} rows received a donor event paragraph whose text coincides with the "
              f"original (the donor row index does differ, so this is not a shuffle bug; usually a "
              f"placeholder sentence such as 'no usable news').")

    out.to_parquet(args.dst)
    print(f"Event-paragraph shuffle done: {stats['n_touched']}/{stats['n_test_rows']} test rows replaced"
          f" ({stats['n_paragraph_mismatch']} rows skipped due to an unexpected paragraph count) -> {args.dst}")
    print(f"Check passed: {n_actually_changed} rows confirmed to have a different event paragraph.")


if __name__ == "__main__":
    main()
