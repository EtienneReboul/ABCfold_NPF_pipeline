"""
scripts/ligand_assignment.py
===============================
NPF protein -> co-folded ligand assignment. Single source of truth for the
apoform -> holoform ligand lists worflows/preprocessing/Snakefile uses to
decide which proteins get which holoform ligand — factored out here so it's
importable (a Snakefile isn't a plain module) by anything else that needs
the same protein/ligand groupings without hand-copying the lists a third
time (they were already duplicated once, into
generate_notebook_protein_cells.py's generated notebook-cell template,
which stays a string-embedded copy by design — notebooks must stay
self-contained). rescoring/src/fit_pocket_lda.py imports this directly to
build one-vs-rest labels per ligand category.

Based on the NPF substrate table (Nitrate transport / Other substrates
columns). High-confidence gibberellin importers get GA1. Everything else
gets the best non-GA ligand supported by the table — nitrate preferred
(simplest, and "Yes" nitrate transport is well documented), ABA as
fallback — disregarding any GA entries in the "Other substrates" column for
non-HC proteins. A handful of remaining proteins get a protein-specific
substrate (auxin, glycerate, dimethylarsenate, JA-Ile, dipeptide,
flavonoid, polyamine) when that's the only ligand the table (or targeted
follow-up literature) documents for them. Two GA-only proteins get GA1 as
a low-confidence guess (LOW_CONFIDENCE_GA_IMPORTERS). NPFs with no
substrate data anywhere are run apoform rather than guessing.
"""

HC_IMPORTERS = [
    "NPF3.1", "NPF4.1", "NPF2.12", "NPF2.13", "NPF2.10", "NPF2.5",
    # medium confidence — not yet included:
    # "NPF2.7", "NPF2.3", "NPF2.4", "NPF4.2", "NPF1.1", "NPF1.2",
]

NITRATE_TRANSPORTERS = [
    "NPF1.1", "NPF1.2", "NPF1.3", "NPF2.3", "NPF2.4", "NPF2.7", "NPF2.9",
    "NPF2.11", "NPF4.6", "NPF5.5", "NPF5.8", "NPF5.9", "NPF5.10", "NPF5.11",
    "NPF5.12", "NPF5.14", "NPF5.16", "NPF6.2", "NPF6.3", "NPF7.2", "NPF7.3",
    "NPF8.5",
]

ABA_TRANSPORTERS = [
    "NPF2.14", "NPF4.2", "NPF4.5", "NPF4.7", "NPF5.1", "NPF5.2", "NPF5.3",
    "NPF5.7",
]

AUXIN_TRANSPORTERS = ["NPF7.1"]              # IAA
GLYCERATE_TRANSPORTERS = ["NPF8.4"]          # Glycerate
DIMETHYLARSENATE_TRANSPORTERS = ["NPF8.1", "NPF8.2"]  # also list peptides, not modeled
JA_ILE_TRANSPORTERS = ["NPF2.6"]             # GA disregarded; JA-Ile is the other substrate
# Gly-Gly: reference dipeptide substrate in Chiang, Stacey & Tsay 2004
# (doi:10.1074/jbc.M405192200) — used there to establish AtPTR2/NPF8.3 as a
# peptide (not nitrate) transporter, and as the 100% baseline for all other
# dipeptides tested.
DIPEPTIDE_TRANSPORTERS = ["NPF8.3"]
# Quercetin-3-O-sophoroside: one of the two pollen-surface flavonol
# diglycosides identified for FST1 (Grunewald et al. 2020, doi:10.1105/tpc.19.00801)
FLAVONOID_TRANSPORTERS = ["NPF2.8"]
# NRT1.3/NPF6.4 mutants show altered spermidine/putrescine resistance and
# uptake (older literature, not in the substrate table); spermidine chosen
# over putrescine/nitrate since it's the substrate directly assayed for
# transport (not just resistance).
POLYAMINE_TRANSPORTERS = ["NPF6.4"]

# Low-tier-confidence GA importers: NPF2.1 and NPF5.6 have GA listed
# directly in the table's "Other substrates" column (disregarded for the
# nitrate/ABA/etc. tiers above, but reinstated here since nothing else
# claimed them). Everything else with zero substrate data in the table
# (NPF2.2, NPF4.3, NPF4.4, NPF5.4, NPF5.13, NPF5.15, NPF6.1) is run apoform
# rather than guessing GA1 from subfamily membership alone.
LOW_CONFIDENCE_GA_IMPORTERS = ["NPF2.1", "NPF5.6"]

# Gibberellin (GA1) importers — the priority group for submit_abcfold.sh's
# manifest ordering.
GIBBERELLIN_IMPORTERS = HC_IMPORTERS + LOW_CONFIDENCE_GA_IMPORTERS

# ligand key (config.yaml's ligands: dict) -> the protein-name list assigned
# that ligand, in ligand_for()'s own precedence order. Single source of
# truth for both ligand_for() below and rescoring/src/fit_pocket_lda.py's
# per-category one-vs-rest label construction.
LIGAND_GROUPS = {
    "GA1": GIBBERELLIN_IMPORTERS,
    "nitrate": NITRATE_TRANSPORTERS,
    "ABA": ABA_TRANSPORTERS,
    "auxin": AUXIN_TRANSPORTERS,
    "glycerate": GLYCERATE_TRANSPORTERS,
    "dimethylarsenate": DIMETHYLARSENATE_TRANSPORTERS,
    "glycylglycine": DIPEPTIDE_TRANSPORTERS,
    "quercetin-3-O-sophoroside": FLAVONOID_TRANSPORTERS,
    "spermidine": POLYAMINE_TRANSPORTERS,
    "JA-Ile": JA_ILE_TRANSPORTERS,
}


def ligand_for(protein_name):
    """NPF name (e.g. 'NPF3.1') is the fasta basename up to the last '_'."""
    npf_name = protein_name.rsplit("_", 1)[0]
    if npf_name in HC_IMPORTERS:
        return "GA1"
    if npf_name in NITRATE_TRANSPORTERS:
        return "nitrate"
    if npf_name in ABA_TRANSPORTERS:
        return "ABA"
    if npf_name in AUXIN_TRANSPORTERS:
        return "auxin"
    if npf_name in GLYCERATE_TRANSPORTERS:
        return "glycerate"
    if npf_name in DIMETHYLARSENATE_TRANSPORTERS:
        return "dimethylarsenate"
    if npf_name in DIPEPTIDE_TRANSPORTERS:
        return "glycylglycine"
    if npf_name in FLAVONOID_TRANSPORTERS:
        return "quercetin-3-O-sophoroside"
    if npf_name in POLYAMINE_TRANSPORTERS:
        return "spermidine"
    if npf_name in JA_ILE_TRANSPORTERS:
        return "JA-Ile"
    if npf_name in LOW_CONFIDENCE_GA_IMPORTERS:
        return "GA1"
    return None


def is_gibberellin_importer(protein_name):
    npf_name = protein_name.rsplit("_", 1)[0]
    return npf_name in GIBBERELLIN_IMPORTERS
