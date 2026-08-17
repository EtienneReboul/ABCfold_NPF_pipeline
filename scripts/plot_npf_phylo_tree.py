#!/usr/bin/env python3
"""
scripts/plot_npf_phylo_tree.py
===============================
Builds a tree from the pipeline's own NPF sequences
(data/sequences/npf_arabidopsis.fasta) and renders it as a customisable
circular dendrogram, with arbitrary named groups of proteins highlighted
(e.g. "gibberellin importers" vs. "glucosinolate importers").

This does NOT reuse the published Fig. 4 tree (Morales de los Rios et al.,
2021, CC BY-NC-ND -- No Derivatives, so re-coloring/re-labeling it would not
be licence-clean). It recomputes a real tree from the same 53 UniProt
sequences already fetched by scripts/download_sequences.py, so any grouping
can be highlighted without touching third-party figures.

Method: MUSCLE v3 end to end -- `muscle -in ... -out ...` for the multiple
alignment, then `muscle -maketree -cluster neighborjoining` (Kimura protein
distance) on that alignment for the tree itself. Both steps are literally
the tool the published figure's caption names ("Alignment ... was carried
out by MUSCLE"); FigTree, which the caption also names, has no alignment or
tree-inference capability of its own -- it is a viewer, so *something* must
have built the tree before FigTree drew it, and MUSCLE's own -maketree is
the parsimonious candidate (same software, no extra step to report). MUSCLE
v3 is deliberately pinned here (not v5): v3 is what a 2021 paper would have
used, and v5 dropped -maketree, alignment only. This still isn't
guaranteed to be the paper's exact pipeline (parameters, MUSCLE version,
or an entirely different tree tool could differ) -- treat the result as
"our own tree of the same proteins, built the way the caption implies",
not a byte-for-byte reproduction of Fig. 4.

--method maps directly to MUSCLE's own -cluster option: neighborjoining
(default; Kimura distance, no molecular-clock assumption -- the standard,
defensible choice and most likely what the paper used), upgma, or upgmb
(both ultrametric -- rounder to draw, every tip on one ring, if the clock
assumption is acceptable for your purposes).

Two earlier versions of this script built a meaningfully worse tree:
  1. Independent pairwise alignments + raw 1-identity distance (no shared
     MSA at all) -- badly saturated once sequences are this diverged
     (median identity here is ~30%), gaps placed inconsistently per pair.
  2. A real MUSCLE MSA, but distances via Biopython's own BLOSUM62
     DistanceCalculator + Biopython's own NJ/UPGMA -- a legitimate
     approach, but a mixed toolchain the paper never claimed, and a plain
     substitution-matrix score isn't a proper corrected evolutionary
     distance the way Kimura protein distance is.
Both produced a poorly-resolved, "starburst" tree in the UPGMA case --
lots of near-simultaneous single branches with no clean nesting. Handing
the whole job to MUSCLE fixes the resolution, not just the rendering.

Rendering: radialtree (github.com/koonimaru/radialtree), not a hand-rolled
matplotlib polar plot. Chosen after benchmarking it against ete3, toytree,
and a custom polar-axis renderer on this same tree -- radialtree drew the
cleanest arcs and its "ring band" leaf-highlighting (a thin colored arc
just outside the tips, one ring per membership slot) reads far better than
stacked dot markers, especially for a leaf in two groups at once. It also
auto-colors major clades for free (scipy's dendrogram color-by-cluster,
which radialtree draws with).

Two real costs, both handled here rather than left as surprises:
  - radialtree only accepts a scipy linkage matrix (from
    scipy.cluster.hierarchy.dendrogram), not a Newick tree, so
    `newick_to_linkage()` converts our MUSCLE tree into one via a
    post-order traversal -- not a re-clustering, the topology and branch
    lengths are preserved exactly.
  - radialtree hasn't been released since 2022 and calls
    `matplotlib.cm.get_cmap`, removed in matplotlib>=3.9. A small shim
    below restores it at import time rather than pinning matplotlib down.

Usage:
    python scripts/plot_npf_phylo_tree.py
    python scripts/plot_npf_phylo_tree.py --groups my_groups.yaml \\
        --output presentation/figures/background/npf_clade_tree.png

    # write a starter YAML for --groups next to the default groups:
    python scripts/plot_npf_phylo_tree.py --dump-groups my_groups.yaml
"""

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch  # pyright: ignore[reportMissingImports]
import yaml  # pyright: ignore[reportMissingImports]
from Bio import Phylo, SeqIO  # pyright: ignore[reportMissingImports]

# Hand-picked clade palette, avoiding two hues in particular: orange (the
# gibberellin highlight ring color) and teal-green (the glucosinolate ring
# color) -- a stock qualitative colormap has no idea those are already
# spoken for. Also sidesteps a real matplotlib quirk: resampling a
# *discrete* qualitative colormap (e.g. Set1's fixed 9 colors) to a
# different count interpolates BETWEEN entries rather than picking distinct
# ones, so two clades can end up nearly the same color by accident (that's
# what happened with Set1 here: NPF4 and NPF5 both landed on near-identical
# oranges). This list is used verbatim, cycling if there are ever more
# clusters than colors, never interpolated.
CLADE_PALETTE = ["#42A5F5", "#AB47BC", "#EC407A", "#EF5350",
                  "#FDD835", "#26C6DA", "#9CCC65", "#78909C"]


class _FixedPalette:
    """Minimal stand-in for a matplotlib ListedColormap: exposes `.colors`
    (what radialtree's non-LinearSegmentedColormap branch reads) without
    going through any interpolation."""
    def __init__(self, colors):
        self.colors = colors


def _get_cmap(name: str, n: int | None = None):
    if name == "npf_clades":
        return _FixedPalette([CLADE_PALETTE[i % len(CLADE_PALETTE)] for i in range(n or 1)])
    # radialtree (last released 2022) calls the pre-3.9 matplotlib.cm.get_cmap
    # API for any other named colormap; matplotlib.colormaps[...] is its
    # modern replacement.
    return matplotlib.colormaps[name].resampled(n) if n else matplotlib.colormaps[name]


mcm.get_cmap = _get_cmap
import radialtree as rt  # noqa: E402  pyright: ignore[reportMissingImports]

CLUSTER_METHODS = {"nj": "neighborjoining", "upgma": "upgma", "upgmb": "upgmb"}

REPO_ROOT = Path(__file__).resolve().parent.parent
GENE_RE = re.compile(r"GN=(NPF[\d.]+)")

# ── Default highlight groups ─────────────────────────────────────────────────
# Gibberellin importers: data/fold_inputs/priority_gibberellin.txt (the 8
# GA1-ligand proteins already run first through the pipeline).
# Glucosinolate importers: GTR1/GTR2/GTR3 = NPF2.10/NPF2.11/NPF2.9
# (Nour-Eldin et al. 2012; Saito et al. 2015) -- all three sit in the NPF2
# clade, unlike the gibberellin group which is scattered across NPF2/3/4.
DEFAULT_GROUPS = {
    "gibberellin_importers": {
        "label": "Gibberellin importers (weak clade colocalisation)",
        "color": "#D55E00",
        "members": [
            "NPF2.1", "NPF2.5", "NPF2.10", "NPF2.12", "NPF2.13",
            "NPF3.1", "NPF4.1", "NPF5.6",
        ],
    },
    "glucosinolate_importers": {
        "label": "Glucosinolate importers / GTR1-3 (strong clade colocalisation)",
        "color": "#009E73",
        "members": ["NPF2.9", "NPF2.10", "NPF2.11"],
    },
}

# radialtree hardcodes black text (no color param on its ax.text calls) and
# a literal "black" for the backbone -- the "above cluster threshold"
# branches, drawn wherever scipy's dendrogram color-by-cluster falls back to
# its default color "C0". Neither responds to matplotlib rcParams or a
# facecolor, so plot_tree() patches both after the fact: text color via a
# plt.rc_context around the radialTreee() call (which *does* work, since
# ax.text with no explicit color argument resolves from rcParams at call
# time), and backbone color via a post-hoc scan of ax.lines for anything
# that rendered as literal black. "none_ring" is the ring-band color for
# leaves in no group at all -- a neutral track behind the highlighted
# segments, tuned separately per theme so it doesn't glow against a dark bg.
THEMES = {
    "light": dict(bg="white", none_ring="#DDDDDD", label="#111111", backbone="#000000"),
    "transparent": dict(bg="none", none_ring="#586E75", label="#93A1A1", backbone="#93A1A1"),
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fasta", default=str(REPO_ROOT / "data/sequences/npf_arabidopsis.fasta"),
                   help="FASTA with GN=NPFx.y in each header (default: pipeline's own NPF set)")
    p.add_argument("--groups", default=None,
                   help="YAML file of {group: {label, color, members}}; default: built-in "
                        "gibberellin/glucosinolate groups. See --dump-groups.")
    p.add_argument("--dump-groups", metavar="PATH",
                   help="Write the default groups YAML to PATH and exit (edit, then pass via --groups)")
    p.add_argument("--alignment-cache", default=str(REPO_ROOT / "data/sequences/npf_muscle_alignment.afa"),
                   help="Cache the MUSCLE multiple alignment here (skips realigning on reruns)")
    p.add_argument("--tree-cache", default=None,
                   help="Cache the MUSCLE Newick tree here (default: "
                        "data/sequences/npf_muscle_tree_<method>.nwk, method-specific since the "
                        "cluster algorithm changes the result)")
    p.add_argument("--output", default=str(REPO_ROOT / "presentation/figures/background/npf_clade_tree.png"),
                   help="Output image path (.png/.pdf)")
    p.add_argument("--method", choices=list(CLUSTER_METHODS), default="upgma",
                   help="Passed straight through to MUSCLE's -cluster option. upgma: ultrametric, "
                        "every tip on one ring -- rounder, easier to read on a slide (default). "
                        "nj: neighbor joining, Kimura distance, no molecular-clock assumption -- "
                        "the more defensible choice if branch length must be trusted; also the "
                        "most likely candidate for what the published figure actually used. "
                        "upgmb: like upgma with max instead of average linkage.")
    p.add_argument("--theme", choices=list(THEMES), default="light",
                   help="light: white background, black text/backbone (default). transparent: "
                        "no background fill, light text/backbone tuned for a dark slide -- pairs "
                        "with --output *.png, save with matplotlib's transparent=True so the "
                        "alpha channel survives.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--figsize", type=float, default=13.0, help="Square figure side, inches")
    p.add_argument("--label-fontsize", type=float, default=8.0)
    p.add_argument("--legend-fontsize", type=float, default=None,
                   help="Legend text size (default: label-fontsize * 0.85 -- scales with it so "
                        "the legend doesn't shrink to unreadable once the whole figure gets "
                        "scaled down to fit a slide)")
    p.add_argument("--palette", default="npf_clades",
                   help="Color source for auto-detected clade branches (scipy's dendrogram "
                        "color-by-cluster). Default 'npf_clades' is this script's own hand-picked "
                        "palette (see CLADE_PALETTE) -- readable on a dark background and "
                        "deliberately avoids orange/teal, already used by the highlight-group "
                        "rings. Any matplotlib colormap name also works, e.g. 'Set1'; pass '' to "
                        "disable and draw every branch black.")
    p.add_argument("--title", default=None)
    p.add_argument("--no-legend", action="store_true",
                   help="Omit the highlighted-groups legend (e.g. when the caller already has a "
                        "color key, as the presentation slide's bullet text does)")
    p.add_argument("--no-clade-legend", action="store_true",
                   help="Omit the auto-detected-clade color legend")
    return p.parse_args()


# ── Sequences, MUSCLE alignment & MUSCLE tree ───────────────────────────────────

def load_sequences_by_gene(fasta_path: Path) -> dict[str, str]:
    seqs = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        m = GENE_RE.search(record.description)
        if not m:
            continue
        seqs[m.group(1)] = str(record.seq)
    if not seqs:
        raise RuntimeError(f"No 'GN=NPFx.y' headers found in {fasta_path}")
    return seqs


def _npf_sort_key(name: str):
    m = re.match(r"NPF(\d+)\.(\d+)", name)
    return (int(m.group(1)), int(m.group(2))) if m else (99, 0)


def _find_muscle3() -> str:
    muscle_bin = shutil.which("muscle")
    if muscle_bin is None:
        raise RuntimeError(
            "muscle not found on PATH. Install it (it's in envs/phylo_tree.yaml, "
            "bioconda channel) and `conda activate npf-phylo-tree`."
        )
    version = subprocess.run([muscle_bin, "-version"], capture_output=True, text=True).stdout
    if "MUSCLE v3" not in version:
        raise RuntimeError(
            f"This script needs MUSCLE v3 (for -maketree; got: {version.strip()!r}). "
            "MUSCLE v5 only aligns -- pin muscle=3.8.1551 in envs/phylo_tree.yaml."
        )
    return muscle_bin


def get_or_build_alignment(seqs: dict[str, str], cache_path: Path) -> Path:
    """Returns the path to an aligned FASTA, aligning with MUSCLE v3 if no
    usable cache exists. Record ids are the NPFx.y gene names."""
    expected = set(seqs)
    if cache_path.exists():
        aligned_ids = {r.id for r in SeqIO.parse(cache_path, "fasta")}
        if aligned_ids == expected:
            print(f"[phylo] Loaded cached MUSCLE alignment from {cache_path}")
            return cache_path
        print("[phylo] Cached alignment covers a different sequence set -- realigning.")

    muscle_bin = _find_muscle3()
    names = sorted(seqs, key=_npf_sort_key)
    with tempfile.TemporaryDirectory() as tmp:
        in_fasta = Path(tmp) / "npf_input.fasta"
        out_fasta = Path(tmp) / "npf_aligned.afa"
        in_fasta.write_text("".join(f">{name}\n{seqs[name]}\n" for name in names))

        print(f"[phylo] Aligning {len(names)} sequences with MUSCLE v3 ...")
        subprocess.run([muscle_bin, "-in", str(in_fasta), "-out", str(out_fasta)],
                        check=True, capture_output=True, text=True)

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(out_fasta, cache_path)

    print(f"[phylo] Cached MUSCLE alignment -> {cache_path}")
    return cache_path


def get_or_build_tree(aligned_fasta: Path, cache_path: Path, method: str):
    """Returns a Bio.Phylo tree, built with MUSCLE v3's own -maketree (Kimura
    protein distance) if no usable cache exists -- not a Biopython-side
    reconstruction, so the whole pipeline is the one tool the paper names."""
    if cache_path.exists():
        print(f"[phylo] Loaded cached MUSCLE tree from {cache_path}")
    else:
        muscle_bin = _find_muscle3()
        cluster = CLUSTER_METHODS[method]
        print(f"[phylo] Building tree with MUSCLE -maketree -cluster {cluster} ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([muscle_bin, "-maketree", "-in", str(aligned_fasta),
                         "-out", str(cache_path), "-cluster", cluster],
                        check=True, capture_output=True, text=True)
        print(f"[phylo] Cached MUSCLE tree -> {cache_path}")

    tree = Phylo.read(cache_path, "newick")
    if method == "nj":
        tree.root_at_midpoint()
    # MUSCLE's NJ can emit tiny negative branch lengths (well-known NJ
    # artifact); clip so radii stay monotonic outward from the root.
    for clade in tree.find_clades():
        if clade.branch_length is not None and clade.branch_length < 0:
            clade.branch_length = 0.0
    return tree


def newick_to_linkage(tree) -> tuple[np.ndarray, list[str]]:
    """Converts the (ultrametric) Bio.Phylo tree into a scipy linkage matrix
    + leaf label order -- radialtree only accepts scipy dendrogram output,
    not Newick, so this preserves our MUSCLE tree's exact topology and
    branch lengths rather than re-clustering from scratch.

    Explicit post-order recursion (not a sort-by-height pass): children
    always get a cluster id assigned before their parent needs it, even
    when a branch length is exactly 0 (a numeric sort can't break ties
    consistently, post-order recursion structurally can't get it wrong).
    """
    depths = tree.depths(unit_branch_lengths=False)
    leaves = tree.get_terminals()
    leaf_order = [c.name for c in leaves]
    R = max(depths[c] for c in leaves)

    cluster_id = {c: i for i, c in enumerate(leaves)}
    counts = {c: 1 for c in leaves}
    rows = []
    next_id = [len(leaves)]

    def visit(node):
        if node.is_terminal():
            return
        for child in node.clades:
            visit(child)
        height = round(R - depths[node], 8)
        cur, cur_n = cluster_id[node.clades[0]], counts[node.clades[0]]
        for child in node.clades[1:]:
            other, other_n = cluster_id[child], counts[child]
            rows.append([cur, other, height, cur_n + other_n])
            cur, cur_n = next_id[0], cur_n + other_n
            next_id[0] += 1
        cluster_id[node] = cur
        counts[node] = cur_n

    visit(tree.root)
    return np.array(rows, dtype=float), leaf_order


# ── Plotting ─────────────────────────────────────────────────────────────────────

def _clade_color_map(color_codes: list[str], palette: str) -> dict[str, object]:
    """Reproduces radialtree's own color_code -> RGBA resolution exactly (see
    radialTreee(): `ucolors = sorted(set(color_list))` then samples `palette`
    at len(ucolors) points), so a legend built from this matches what's
    actually drawn pixel for pixel."""
    ucolors = sorted(set(color_codes))
    if not palette:
        return {c: "black" for c in ucolors}
    cmp = mcm.get_cmap(palette, len(ucolors))
    cmap_colors = (cmp(np.linspace(0, 1, len(ucolors)))
                   if isinstance(cmp, matplotlib.colors.LinearSegmentedColormap)
                   else cmp.colors)
    return {c: cmap_colors[i] for i, c in enumerate(ucolors)}


def _clade_label(names: list[str]) -> str:
    """'NPF2.1-2.7 (7)' for a clean single-subfamily cluster (the .x-.y
    range disambiguates when one subfamily got split into two separate
    auto-detected clusters, e.g. NPF2 here), 'NPF1/6/7/8 (11)' for one of
    scipy's automatic distance-threshold clusters that happens to span
    several NPF subfamily numbers instead."""
    parsed = sorted((int(n.split(".")[0].replace("NPF", "")), int(n.split(".")[1])) for n in names)
    prefixes = sorted({p for p, _ in parsed})
    if len(prefixes) == 1:
        p = prefixes[0]
        suffixes = [s for pp, s in parsed if pp == p]
        return f"NPF{p}.{suffixes[0]}-{p}.{suffixes[-1]} ({len(names)})"
    return f"NPF{'/'.join(str(p) for p in prefixes)} ({len(names)})"


def plot_tree(tree, groups: dict, theme: dict, label_fontsize: float, figsize: float,
              palette: str, title: str | None, show_legend: bool = True,
              show_clade_legend: bool = True, legend_fontsize: float | None = None):
    legend_fontsize = legend_fontsize or label_fontsize * 0.85
    Z, leaf_order = newick_to_linkage(tree)
    Z2 = sch.dendrogram(Z, labels=leaf_order, no_plot=True)
    ordered_labels = Z2["ivl"]  # leaf order after scipy's own dendrogram ordering

    # gene -> [group_key, ...] it belongs to, in a stable order
    membership: dict[str, list[str]] = {}
    for gkey, gdef in groups.items():
        for gene in gdef["members"]:
            membership.setdefault(gene, []).append(gkey)

    # One ring band per membership "slot" (not per group): ring 0 is every
    # leaf's first group, or the neutral grey if it has none; ring 1 is a
    # second group if a leaf happens to belong to two, else fully
    # transparent. This is what makes a doubly-highlighted leaf (e.g. one
    # importer counted in both groups) show as two stacked ring segments
    # instead of just picking one color to show.
    max_slots = max((len(g) for g in membership.values()), default=0)
    max_slots = max(max_slots, 1)

    def slot_color(gene: str, slot: int):
        my_groups = membership.get(gene, [])
        if slot < len(my_groups):
            return groups[my_groups[slot]]["color"]
        return theme["none_ring"] if slot == 0 else (1, 1, 1, 0)

    def to_rgba(c):
        if isinstance(c, tuple):
            return c
        h = c.lstrip("#")
        r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
        return (r, g, b, 1.0)

    colorlabels = {
        f"slot{slot}": [to_rgba(slot_color(name, slot)) for name in ordered_labels]
        for slot in range(max_slots)
    }

    fig, ax = plt.subplots(figsize=(figsize, figsize), facecolor=theme["bg"])
    ax.set_facecolor(theme["bg"])
    # ax.text() with no explicit color resolves from rcParams at call time,
    # which is the only lever radialtree exposes for label color.
    with plt.rc_context({"text.color": theme["label"]}):
        rt.radialTreee(Z2, fontsize=label_fontsize, ax=ax,
                       pallete=palette or None, colorlabels=colorlabels)
    ax.set_aspect("equal")

    # radialtree hardcodes the backbone (above-cluster-threshold branches)
    # to the literal string "black" with no override -- swap it after the
    # fact for anything that isn't the light theme.
    for line in ax.get_lines():
        c = line.get_color()
        if isinstance(c, str) and c in ("black", "k", "#000000"):
            line.set_color(theme["backbone"])

    # radialTreee sets xlim/ylim tight around the circle in DATA coordinates
    # that never account for fontsize (no built-in margin for a legend, or
    # even for its own labels at large fontsize), so anchor the legend below
    # axes fraction 0 rather than inside it -- bbox_inches="tight" on save
    # expands the canvas to fit. How far below scales with label_fontsize:
    # bigger leaf labels overhang the fixed-radius circle by more in display
    # space, so a legend anchored at a fixed offset collides with them once
    # labels get large (this is tuned empirically, not derived). Group-
    # highlight and clade legends sit side by side rather than stacked so
    # two short columns don't turn into one tall one that needs even more
    # clearance below the circle.
    legend_y = -(0.04 + label_fontsize * 0.01)
    group_legend = None
    if show_legend:
        handles = [
            plt.Line2D([0], [0], marker="s", linestyle="", markersize=9,
                       color=gdef["color"], label=gdef["label"])
            for gdef in groups.values()
        ]
        anchor_x = 0.27 if show_clade_legend else 0.5
        group_legend = ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(anchor_x, legend_y),
                                  frameon=False, fontsize=legend_fontsize, ncol=1, title="Highlighted groups",
                                  title_fontsize=legend_fontsize, alignment="left")
        for text in list(group_legend.get_texts()) + [group_legend.get_title()]:
            text.set_color(theme["label"])

    if show_clade_legend:
        if group_legend is not None:
            ax.add_artist(group_legend)
        leaf_clade_colors = Z2["leaves_color_list"]
        color_map = _clade_color_map(leaf_clade_colors, palette)
        clades: dict[str, list[str]] = {}
        for name, code in zip(ordered_labels, leaf_clade_colors):
            if code == "C0":  # radialtree always forces this one to black (unclustered backbone)
                continue
            clades.setdefault(code, []).append(name)
        ordered_codes = sorted(clades, key=lambda code: min(
            int(n.split(".")[0].replace("NPF", "")) for n in clades[code]))
        clade_handles = [
            plt.Line2D([0], [0], color=color_map[code], lw=2.5, label=_clade_label(clades[code]))
            for code in ordered_codes
        ]
        anchor_x = 0.73 if show_legend else 0.5
        clade_legend = ax.legend(handles=clade_handles, loc="upper center", bbox_to_anchor=(anchor_x, legend_y),
                                  frameon=False, fontsize=legend_fontsize, ncol=1, title="Clades (auto-detected)",
                                  title_fontsize=legend_fontsize, alignment="left")
        for text in list(clade_legend.get_texts()) + [clade_legend.get_title()]:
            text.set_color(theme["label"])

    if title:
        ax.set_title(title, color=theme["label"], fontsize=13, pad=20)

    fig.tight_layout()
    return fig


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.dump_groups:
        Path(args.dump_groups).write_text(yaml.dump(DEFAULT_GROUPS, sort_keys=False))
        print(f"[phylo] Wrote default groups -> {args.dump_groups}")
        return

    groups = DEFAULT_GROUPS
    if args.groups:
        groups = yaml.safe_load(Path(args.groups).read_text())

    seqs = load_sequences_by_gene(Path(args.fasta))
    print(f"[phylo] {len(seqs)} sequences loaded from {args.fasta}")

    aligned_fasta = get_or_build_alignment(seqs, Path(args.alignment_cache))
    tree_cache = Path(args.tree_cache) if args.tree_cache else \
        REPO_ROOT / f"data/sequences/npf_muscle_tree_{args.method}.nwk"
    tree = get_or_build_tree(aligned_fasta, tree_cache, args.method)

    theme = THEMES[args.theme]
    fig = plot_tree(tree, groups, theme, args.label_fontsize, args.figsize,
                     args.palette, args.title, show_legend=not args.no_legend,
                     show_clade_legend=not args.no_clade_legend,
                     legend_fontsize=args.legend_fontsize)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if theme["bg"] == "none":
        fig.savefig(out, dpi=args.dpi, transparent=True, bbox_inches="tight")
    else:
        fig.savefig(out, dpi=args.dpi, facecolor=theme["bg"], bbox_inches="tight")
    print(f"[phylo] Wrote {out}")


if __name__ == "__main__":
    main()
