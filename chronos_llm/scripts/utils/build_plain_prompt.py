"""Plain prompt: reduce the forecast prompt to a plain prompt that drops the reasoning
instructions, keeps every fact, and reads as fluent prose.

Current prompt template (identical across datasets):
  [instruction block "Your task is to infer..."] + ## Dataset background + ## Event (established fact)
  + ## Series metadata + ## Task [reasoning instruction + <think> format]
Reduction = delete the leading and trailing instruction blocks, keep the three factual sections
(background / event / series metadata), drop the markdown headings, turn the metadata bullets into
a sentence, and join into fluent paragraphs. The information content equals the original prompt.
"""
import re


def _parse_sections(prompt):
    # Split on line-initial "## "; parts[0] is the untitled leading instruction block (discarded)
    parts = re.split(r'\n##+ ', '\n' + prompt)
    sec = {}
    for p in parts[1:]:
        title, _, body = p.partition('\n')
        sec[title.strip()] = body.strip()
    return sec


def _metadata_to_sentence(meta):
    """Turn the bulleted series metadata into a fluent sentence."""
    d = {}
    for line in meta.splitlines():
        m = re.match(r'\s*-\s*([^:]+):\s*(.+)', line)
        if m:
            d[m.group(1).strip().lower()] = m.group(2).strip()
    freq = d.get('frequency')
    hw, fw = d.get('history window', ''), d.get('future window', '')
    out = []
    if freq:
        out.append(f"The series is sampled every {freq}")
    if hw:
        out.append(f"the history window covers {hw}")
    if fw:
        out.append(f"and the future window to forecast covers {fw}")
    return (", ".join(out) + ".") if out else ""


def simplify_prompt(prompt):
    sec = _parse_sections(prompt)
    bg = sec.get('Dataset background', '').strip()
    # The event heading may be "Event (established fact)" or just "Event"
    ev = next((v for k, v in sec.items() if k.lower().startswith('event')), '').strip()
    meta = sec.get('Series metadata', '')
    meta_sent = _metadata_to_sentence(meta)
    ev_txt = ev if ev.lower().startswith(('the ', 'prediction', 'on ')) else ev
    paras = [bg]
    if ev_txt:
        paras.append(ev_txt)
    if meta_sent:
        paras.append(meta_sent)
    # Soft line breaks inside the background/event sections become spaces; paragraphs are
    # separated by blank lines => fluent prose
    paras = [re.sub(r'\s*\n\s*', ' ', p).strip() for p in paras if p]
    return "\n\n".join(paras)


def add_plain_prompt_column(parquet_path, output=None):
    """Add a plain_prompt column (= simplify_prompt(prompt)) to the parquet, leaving the rest of
    the schema unchanged."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path)
    prompts = t.column("prompt").to_pylist()
    plain = pa.array([simplify_prompt(p) for p in prompts], type=pa.string())
    if "plain_prompt" in t.schema.names:
        t = t.set_column(t.schema.get_field_index("plain_prompt"), "plain_prompt", plain)
    else:
        t = t.append_column("plain_prompt", plain)
    pq.write_table(t, output or parquet_path)
    print(f"wrote {output or parquet_path}: {t.num_rows} rows, plain_prompt column added")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="data/forecast/mmtr_forecast_corpus.parquet")
    ap.add_argument("--apply", action="store_true", help="write the plain_prompt column back; otherwise only show examples")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.apply:
        add_plain_prompt_column(args.parquet, args.output)
    else:
        import pyarrow.parquet as pq
        df = pq.read_table(args.parquet, columns=['dataset_name', 'prompt']).to_pandas()
        for d in ['fnf/load', 'CGTSF/MSPG']:
            p = df[df['dataset_name'] == d]['prompt'].iloc[0]
            print(f"\n{'='*90}\n### {d}  -- simplified plain prompt:\n{'='*90}")
            print(simplify_prompt(p))
