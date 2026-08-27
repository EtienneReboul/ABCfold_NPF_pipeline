#!/usr/bin/env python3
"""
rescoring/src/build_position_mapping.py
==========================================
Reconstruct, for every protein in this pipeline's corpus, the per-protein
residue number (`resnr`, in the same full-sequence numbering ABCfold/PLIP
use) at each of the 35 "position" indices `NPF_LDA_kernel` uses throughout
its pocket/LDA outputs.

Generalized from the sibling NPF_pocket_pipeline/rescoring project's version
of this script, which covered only 33 hand-picked ("hc") proteins snapshot-
copied out of NPF_LDA_kernel. Confirmed by direct inspection: NPF_LDA_kernel's
own `data/cdd_msa/npf_aligned.sto` (HMMER alignment) and
`results/ga_classifier/pocket_sites_cdd_msa.tsv` already cover, 1:1, every
one of this pipeline's 53 `data/sequences/*.fasta` proteins — so there's no
new bioinformatics needed here, just reading NPF_LDA_kernel's files directly
(never written to) instead of the hc-only snapshot the sibling project
copied in.

Why this is needed at all: "position" 1-35 in NPF_LDA_kernel's outputs is a
1-based ascending **Stockholm-alignment-column index** (cd17351/MFS_NPF
profile HMM), NOT a raw ascending residue number — two different proteins'
residue 158 is not necessarily "the same" pocket position, since indels
shift the correspondence protein to protein.

Method
------
1. Load the cached Stockholm alignment (npf_aligned.sto).
2. Map the anchor protein's (NPF6.1_Q9LYR6, the corpus's one cd17351
   root-entry match) own binding-site residues to alignment columns — this
   reproduces the exact 35 columns ("positions") NPF_LDA_kernel's own
   extract_cdd_msa.py uses.
3. For every protein in this pipeline's corpus, invert the alignment at
   those 35 columns to get its own resnr (or NaN if that protein has a gap
   at that column).
4. Validate: re-derive each protein's 35-letter pocket string from its own
   FASTA + the reconstructed resnr and assert it matches NPF_LDA_kernel's
   own pocket_sites_cdd_msa.tsv row exactly. Fails loudly on any mismatch.

Output: data/position_resnr_map.csv (protein, position, resnr)

**`--full` mode** (2026-08-26, at the user's request): the CDD-only 35
positions above are a curated SUBSET of `npf_aligned.sto`'s own columns --
that Stockholm alignment already spans the full protein (746 columns,
confirmed by hand: covers all 53 corpus proteins end to end, not just the
domain window around the pocket), so no new TM-helix realignment is
actually needed to get a position index that works globally across every
protein -- the existing, already-validated sequence alignment already
provides one for free. `--full` maps EVERY alignment column (not just the
35 CDD ones) to each protein's resnr, so a Rosetta-contacted residue
OUTSIDE the CDD-annotated pocket still gets a comparable cross-protein
"position" instead of being silently dropped -- see
`rescore_redocked_aggregate.py`'s `add_position()`/`scan_position_cohesion.py`
for where this gets used. Output: data/position_resnr_map_full.csv
(protein, position [1-746, alignment-column-based], resnr, is_cdd_pocket).

**Caveat, read before trusting a `--full`-mode result outside the pocket**:
this is a sequence-only alignment (HMMER/CDD profile-based), not a
structural (3D) one -- correspondence quality is expected to degrade in
poorly-conserved loop regions and can, in principle, misalign individual
TM helices relative to a true structural superposition when insertions/
deletions between paralogs shift the register. The 35 CDD positions are
validated indirectly (they reproduce NPF_LDA_kernel's own published
pocket strings exactly); the remaining ~700 columns are NOT independently
validated the same way -- treat a `--full`-mode hit outside the pocket as
a lead worth checking against the actual 3D structure (e.g. is the
residue really in the same TM helix / spatial neighborhood across
proteins), not a result as solid as a CDD-position hit. A true structural
realignment (TM-align across representative structures, extending
`scripts/tm_helix_alignment.py`'s existing per-protein TM-Ca extraction)
would be the rigorous fix if this sequence-based map turns out noisy in
practice -- not attempted here, flagged as a possible follow-up.
"""
import re
import sys

from Bio import AlignIO

import config

_UNIPROT_RE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})\b"
)


def extract_uid(header: str) -> str | None:
    m = _UNIPROT_RE.search(header)
    return m.group(0) if m else None


def seq_pos_to_cols(gapped_seq: str, target_positions: set[int]) -> list[int]:
    """1-based ungapped seq positions -> sorted 0-based alignment column indices."""
    cols, seq_pos = [], 0
    for col, char in enumerate(gapped_seq):
        if char not in ("-", "."):
            seq_pos += 1
            if seq_pos in target_positions:
                cols.append(col)
    return sorted(cols)


def cols_to_seq_pos(gapped_seq: str, cols: list[int]) -> dict[int, int | None]:
    """0-based alignment column -> 1-based ungapped seq position (None if gap there)."""
    col_set = set(cols)
    out: dict[int, int | None] = {}
    seq_pos = 0
    for col, char in enumerate(gapped_seq):
        is_gap = char in ("-", ".")
        if not is_gap:
            seq_pos += 1
        if col in col_set:
            out[col] = None if is_gap else seq_pos
    return out


def load_alignment() -> tuple[dict[str, str], dict[str, str]]:
    """Returns (uid -> gapped_seq, uid -> pipeline_name)."""
    if not config.NPF_LDA_KERNEL_ALIGNMENT.exists():
        sys.exit(f"{config.NPF_LDA_KERNEL_ALIGNMENT} not found -- is NPF_LDA_kernel "
                  "checked out as a sibling of this pipeline?")
    aln = AlignIO.read(str(config.NPF_LDA_KERNEL_ALIGNMENT), "stockholm")
    uid_to_seq, uid_to_name = {}, {}
    for rec in aln:
        header = f"{rec.id} {rec.description}"
        uid = extract_uid(header)
        if uid is None:
            continue
        uid_to_seq[uid] = str(rec.seq)
        m = re.search(r"GN=(\S+)", header)
        if m:
            uid_to_name[uid] = f"{m.group(1)}_{uid}"
    return uid_to_seq, uid_to_name


def corpus_protein_names() -> list[str]:
    """Every base protein this pipeline knows about, from
    data/sequences/sequences.done (same source
    worflows/postprocessing/Snakefile's PROTEINS reads)."""
    if not config.SEQ_SENTINEL.exists():
        sys.exit(f"{config.SEQ_SENTINEL} not found -- run "
                  "worflows/preprocessing/Snakefile first.")
    return [line.strip() for line in config.SEQ_SENTINEL.read_text().splitlines() if line.strip()]


def load_fasta(protein: str) -> str:
    text = (config.SEQUENCES_DIR / f"{protein}.fasta").read_text()
    return "".join(l.strip() for l in text.splitlines() if not l.startswith(">"))


def load_expected_pockets() -> dict[str, str]:
    if not config.NPF_LDA_KERNEL_POCKET_SITES.exists():
        sys.exit(f"{config.NPF_LDA_KERNEL_POCKET_SITES} not found.")
    expected = {}
    for line in config.NPF_LDA_KERNEL_POCKET_SITES.read_text().splitlines()[1:]:
        if not line.strip():
            continue
        name, pocket = line.split("\t")
        expected[name] = pocket
    return expected


def _anchor_cols(uid_to_seq: dict[str, str]) -> list[int]:
    if config.ANCHOR_UNIPROT_ID not in uid_to_seq:
        sys.exit(f"Anchor {config.ANCHOR_UNIPROT_ID} not found in alignment.")
    if not config.NPF_LDA_KERNEL_ANCHOR_BINDING_SITE.exists():
        sys.exit(f"{config.NPF_LDA_KERNEL_ANCHOR_BINDING_SITE} not found.")
    anchor_resnr = sorted(
        int(x)
        for x in config.NPF_LDA_KERNEL_ANCHOR_BINDING_SITE.read_text().strip().split(",")
    )
    best_cols = seq_pos_to_cols(uid_to_seq[config.ANCHOR_UNIPROT_ID], set(anchor_resnr))
    if len(best_cols) != len(anchor_resnr):
        sys.exit(
            f"Anchor mapping incomplete: {len(best_cols)}/{len(anchor_resnr)} "
            "binding-site residues mapped to alignment columns."
        )
    print(f"[build_position_mapping] {len(best_cols)} positions anchored on {config.ANCHOR_PROTEIN}")
    return best_cols


def main_cdd_only():
    uid_to_seq, _ = load_alignment()
    best_cols = _anchor_cols(uid_to_seq)

    names = corpus_protein_names()
    expected_pockets = load_expected_pockets()

    rows = []
    n_ok, n_bad, n_missing = 0, 0, 0
    for name in names:
        uid = name.split("_")[-1]
        if uid not in uid_to_seq:
            print(f"  [!] {name}: not in Stockholm alignment -- skipped", file=sys.stderr)
            n_missing += 1
            continue

        mapping = cols_to_seq_pos(uid_to_seq[uid], best_cols)
        resnrs = [mapping[c] for c in best_cols]

        seq = load_fasta(name)
        derived_pocket = "".join(seq[r - 1] if r is not None else "X" for r in resnrs)
        expected = expected_pockets.get(name)
        if expected is None:
            sys.exit(f"No expected pocket string for {name} in {config.NPF_LDA_KERNEL_POCKET_SITES}")
        if derived_pocket != expected:
            print(f"  [!] MISMATCH {name}: derived={derived_pocket} expected={expected}", file=sys.stderr)
            n_bad += 1
            continue
        n_ok += 1

        for position, resnr in enumerate(resnrs, start=1):
            rows.append({"protein": name, "position": position, "resnr": resnr})

    print(f"[build_position_mapping] validated {n_ok}/{len(names)} proteins "
          f"({n_bad} mismatched, {n_missing} missing from alignment)")
    if n_bad or n_missing:
        sys.exit("Validation failed for one or more proteins -- see warnings above.")

    import pandas as pd

    df = pd.DataFrame(rows, columns=["protein", "position", "resnr"])
    df.to_csv(config.POSITION_RESNR_MAP_CSV, index=False)
    print(f"[build_position_mapping] wrote {config.POSITION_RESNR_MAP_CSV} "
          f"({len(df)} rows, {df.protein.nunique()} proteins)")


def main_full():
    """Every alignment column (1-746), not just the 35 CDD ones -- see
    module docstring's --full section for the caveats. No pocket-string
    validation here (that's CDD-specific); instead cross-checks that the
    35 CDD columns' resnrs, sliced back out of this full map, exactly
    match main_cdd_only()'s own already-validated output."""
    import pandas as pd

    uid_to_seq, _ = load_alignment()
    best_cols = _anchor_cols(uid_to_seq)  # ordered -- index+1 IS the 1-35 CDD "position" numbering
    best_cols_set = set(best_cols)
    col_to_cdd_position = {col: i + 1 for i, col in enumerate(best_cols)}
    alignment_length = len(next(iter(uid_to_seq.values())))
    all_cols = list(range(alignment_length))

    names = corpus_protein_names()
    rows = []
    n_missing = 0
    for name in names:
        uid = name.split("_")[-1]
        if uid not in uid_to_seq:
            print(f"  [!] {name}: not in Stockholm alignment -- skipped", file=sys.stderr)
            n_missing += 1
            continue
        mapping = cols_to_seq_pos(uid_to_seq[uid], all_cols)
        for position, col in enumerate(all_cols, start=1):
            rows.append({
                "protein": name, "position": position, "resnr": mapping[col],
                "is_cdd_pocket": col in best_cols_set,
                # 1-35 CDD numbering (matches lda_GA1_loadings.tsv/position_importance_GA1.tsv's
                # own "position" column) -- NaN outside the CDD pocket, where those files have no
                # entry anyway. Needed because THIS column's own "position" is the 1-746
                # whole-alignment index, a different numbering scheme entirely.
                "cdd_position": col_to_cdd_position.get(col),
            })

    df = pd.DataFrame(rows, columns=["protein", "position", "resnr", "is_cdd_pocket", "cdd_position"])

    # Cross-check against the already-validated CDD-only map, if it exists.
    if config.POSITION_RESNR_MAP_CSV.exists():
        cdd_only = pd.read_csv(config.POSITION_RESNR_MAP_CSV)
        full_cdd_rows = df[df["is_cdd_pocket"]]
        merged = full_cdd_rows.merge(
            cdd_only, left_on=["protein", "cdd_position"], right_on=["protein", "position"],
            suffixes=("_full", "_cdd"),
        )
        both_nan = merged["resnr_full"].isna() & merged["resnr_cdd"].isna()
        mismatches = merged[(merged["resnr_full"] != merged["resnr_cdd"]) & ~both_nan]
        if len(mismatches):
            sys.exit(f"--full mode's CDD-column subset disagrees with {config.POSITION_RESNR_MAP_CSV} "
                      f"on {len(mismatches)} row(s) -- something is inconsistent, not safe to trust "
                      f"the full map. First mismatch:\n{mismatches.iloc[0]}")
        print(f"[build_position_mapping] --full mode's {len(merged)} CDD-column rows exactly match "
              f"{config.POSITION_RESNR_MAP_CSV} -- consistent.")
    else:
        print(f"[build_position_mapping] NOTE: {config.POSITION_RESNR_MAP_CSV} doesn't exist yet -- "
              "skipping the CDD-subset cross-check (run without --full first to enable it).")

    df.to_csv(config.POSITION_RESNR_MAP_FULL_CSV, index=False)
    print(f"[build_position_mapping] wrote {config.POSITION_RESNR_MAP_FULL_CSV} "
          f"({len(df)} rows, {df.protein.nunique()}/{len(names)} proteins, "
          f"{alignment_length} alignment columns, {n_missing} proteins missing from alignment)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                     help="map every alignment column (746), not just the 35 CDD pocket ones -- see "
                          "module docstring's --full section")
    args = ap.parse_args()
    if args.full:
        main_full()
    else:
        main_cdd_only()
