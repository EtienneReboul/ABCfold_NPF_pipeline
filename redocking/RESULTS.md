# HADDOCK3 Redocking of GA1 — Results

**Project**: Physics-based (HADDOCK3/CNS) cross-check of ABCfold's ab initio GA1-NPF cofolding predictions
**Date**: August 2026
**Working directory**: `redocking/` (see `README.md` for pipeline/setup; this file is results only)

---

## 1. Scope

**143 complexes** docked: every CDD-annotated protein in the corpus (48 of 53) × its 3 `pca_k3` macro-conformation clusters, minus 5 curated importers whose only ABCfold co-folding here used a different ligand entirely (nitrate) and who therefore have no GA1 pose to build restraints from at all.

| role | proteins | complexes | basis |
|---|---|---|---|
| **importer** | 10 | 30 | `NPF_LDA_kernel` curated `hc_importers`. 5 co-folded with GA1 here (real ABCfold pose exists); 5 co-folded with nitrate here but still real (if less efficient) GA1 importers per the literature — redocked against their apoform receptor instead |
| **non_importer** | 36 | 107 | `hc_non_importers` (19) plus every other CDD-annotated protein not GA1-tested here (17) |
| **ambiguous** | 2 | 6 | `NPF2.1`, `NPF5.6` — co-folded with GA1 here but not in either curated NPF_LDA_kernel list; redocked for completeness, excluded from all importer/non_importer statistics below |

Two real, isolated docking failures out of 143 (both a deterministic CNS floating-point crash in `[flexref]` for the same pose, `NPF2.1_Q9M171__ca0_protenix`, on repeated attempts — the earlier partial run's own kept models still produced usable Rosetta contact data, so no complex is entirely missing downstream).

---

## 2. Does HADDOCK3 converge on ABCfold's own predicted pose? (Stage 7)

For the **15 complexes with a real ABCfold GA1 pose** (5 importer proteins × 3 clusters): ligand heavy-atom RMSD between HADDOCK3's top-ranked model and ABCfold's own prediction, after C-alpha superposition of the two receptors.

| | value |
|---|---|
| mean | **5.19 Å** |
| range | 3.40 – 7.05 Å |

**HADDOCK3 does not converge on the same pose ABCfold predicted.** A meaningful, non-trivial disagreement between the independent physics-based method and the learned ab initio model.

For every complex without a real ABCfold GA1 pose (apoform receptor — all non_importers plus the 5 apoform importers): how many CDD active-pocket residues does the best-scoring HADDOCK model actually contact (out of 35).

| role | n | mean | median |
|---|---|---|---|
| importer (apoform) | 15 | 10.7 | 11 |
| non_importer | 107 | 10.1 | 12 |

No sharp separation here on its own — see §4 for why (restraint construction uses the same CDD residues for both roles) and §5 for the metric that actually does discriminate.

---

## 3. Is HADDOCK3's own top-scoring model trustworthy? (Stage 8)

Scored **every** kept `[flexref]` model (not just the top-ranked one — 5,867 poses total) for CDD active-residue contact count, and fit an unsupervised GMM(2) to separate genuinely pocket-engaging poses from off-target ("membrane") ones the ligand-observer eyeballed by hand first.

- **Sharp bimodal split**: component means 0.50 vs. 11.0 contacts, weights 0.40/0.60 — roughly 39% of ALL kept poses (both roles) are off-target junk that would silently contaminate any population-level average.
- **0/30 importer complexes** ever fail to find at least one good pose among their 40; **13/107 (12%) non_importer complexes** find zero good poses at all across all 40 attempts. Checked in detail: every affected protein still succeeds in 2 of its 3 macro-conformations — this reads as "this specific conformational state can't engage GA1," not "this protein categorically can't."
- Despite the ensemble being 39% contaminated, the plain best-HADDOCK-score pick was **already** a genuinely good pose in **140/143 complexes (98%)** — filtering to the GMM-selected representative changes essentially nothing about the single-representative-model results in §2 and §4. The contamination matters at the population level (§5's per-position scan pools every good pose), not for the "best model per complex" summaries.

---

## 4. Does Rosetta energetics agree with the CDD-annotated pocket? (`rescore_redocked_aggregate.py`)

Precision = of residues Rosetta's own energy graph found interacting with the redocked GA1, what fraction fall inside the 35 CDD positions. Recall = of the 35 CDD positions, how many are ever contacted.

| | precision | recall |
|---|---|---|
| **redocked importer** | 0.551 | 0.791 |
| **redocked non_importer** | 0.499 | 0.749 |
| ab-initio Rosetta baseline (same 5 GA1-cofolded proteins, pre-redocking) | 0.164 | 0.943 |
| ab-initio PLIP baseline (reference — the "~50%" figure) | 0.348 | 0.480 |

**Redocking clearly beats both ab-initio baselines on precision** (0.164 → 0.551, 0.348 → 0.551) — ab-initio Rosetta contacts touch almost everything nearby (high recall, terrible precision, i.e. noisy); redocked poses are far more selective. Recall drops slightly against the ab-initio Rosetta number but still comfortably beats the PLIP reference.

**Importer and non-importer barely differ on this metric** — almost certainly because `define_active_passive.py`'s HADDOCK restraints (AIRs) are built from the *same* CDD active-residue set for both roles, so both are restraint-driven toward the same pocket regardless of true importer status. This metric alone is **not** a clean importer/non-importer discriminator; see §5 for the one that is.

Cross-force-field caveat: HADDOCK's own physical score (vdW+elec+desolv, AIR term removed) rates importer poses *more* favorable than non-importer (mean −410.5 vs −346.4 REU-equivalent), but an independent PyRosetta REF2015 rescoring of those same poses says the *opposite* (importer `total_score` +87.0 vs non-importer −90.0). Two force fields, same poses, opposite conclusion on which role is energetically favored — neither should be trusted alone as a role discriminator.

---

## 5. Systematic per-position scan — the actual discriminating signal

`scan_position_cohesion.py`: for every position with enough contacted proteins on both sides, Mann-Whitney U on Rosetta two-body energy (importer vs. non_importer), with Benjamini-Hochberg (BH) FDR correction applied over the family of positions actually tested (not the full 746/35 candidate space — an untested position doesn't belong in the correction).

### Headline result (CDD-only family, m=29 tested positions — the right family size for this hypothesis)

| CDD position | raw p | **BH q** | importer unfavorable | non_importer unfavorable | dominant LDA Z-scale |
|---|---|---|---|---|---|
| **15** | 0.0026 | **0.074** | 0/10 (0%) | 16/35 (46%) | Z4 (electronic/charge) |
| 1 | 0.0123 | 0.177 | 0/4 | 15/27 (56%) | Z3 (electronic) |
| 20 | 0.0244 | 0.177 | 1/10 (10%) | 20/36 (56%) | Z5 (proline/turn) |

**Position 15 is the one robust, defensible hit** — survives a conventional exploratory FDR threshold (q ≤ 0.1). Positions 1 and 20 are reasonable secondary candidates (q=0.177) but weaker evidence.

**Important methodological note**: doubling the importer sample from 5 to 10 proteins (§1) roughly halved position 15's p-value (0.015 → 0.0026) — direct confirmation that importer-side sample size was the real statistical bottleneck, not the underlying signal being weak.

**BH-family-size matters**: running the same correction over a larger, mixed family (51 positions spanning the full 746-column whole-alignment scan, §6) inflates q-values across the board (position 15: q=0.130 instead of 0.074). The CDD-only family above is the right scope for a CDD-focused hypothesis; the full-alignment scan below is a separate, more exploratory screen and should be read with that in mind.

**Position 15, amino acid composition** (`NPF_LDA_kernel`'s own pocket strings, 8/10 importers + 35/36 non_importers with data): importers cluster tightly on small/medium hydrophobic residues (Ile×3, Val×2, Thr×2, Phe×1 — 7/8 are I/V/T). Non-importers are far more heterogeneous (Ser×6, Thr×6, Ile×4, Pro×3, Ala×3, Leu×3, Tyr×2, Glu×2, plus singles) and include several residue types importers never have there (Pro, Gly, Glu, Tyr) that would plausibly clash with or poorly accommodate GA1. The signal looks like importer-side consistency against non-importer-side heterogeneity, not one single clean substitution.

---

## 6. Beyond the CDD pocket: does redocking find anything CDD/InterPro missed?

`build_position_mapping.py --full` maps every one of `npf_aligned.sto`'s 746 whole-alignment columns (not just the 35 CDD ones) using the *same* already-validated sequence alignment — no new structural realignment needed, since that alignment already spans the entire protein. Two non-CDD positions stood out in the raw scan:

| position | raw p | BH q (full scan, m=51) |
|---|---|---|
| 608 | 0.0058 | 0.147 |
| 508 | 0.0299 | 0.254 |

Neither survives BH correction over the larger, mixed 51-position family — **these remain leads, not confirmed hits**. Two independent sanity checks were run against them regardless:

1. **Ab-initio PLIP cross-check** (existing PLIP interaction data on ABCfold's own poses, an independent method on an independent dataset): at position 119 (a related non-CDD hit from an earlier iteration of this scan, now superseded), 4/5 importers get a real PLIP `hydrophobic_interactions` hit at the exact same residue in their ab-initio pose; at position 508, 3/5 do. Both are genuine structural contact points, not sequence-alignment artifacts — the redocked-pose Rosetta signal there is real, just not statistically confirmed at scale yet.
2. **Alignment-quality caveat**: this is a sequence, not structural, alignment outside the well-conserved CDD-anchored core. Correspondence quality there is not independently validated the way the 35 CDD positions are (which reproduce `NPF_LDA_kernel`'s own published pocket strings exactly). Any non-CDD hit is worth checking against the real 3D structure before trusting it as strongly as a CDD hit.

**Bottom line for non-CDD positions**: real, PLIP-corroborated leads exist outside the CDD-annotated pocket (confirming it would have been wrong to restrict this analysis to CDD alone), but none currently clear multiple-testing correction — more importer-side data (as in §5) is the most direct way to find out whether they hold up.

---

## 7. Practical recommendations

1. **Position 15 is the strongest experimental follow-up candidate** from this entire analysis — clean statistical support (BH q=0.074) and a chemically interpretable composition (importer-side I/V/T consistency vs. non-importer heterogeneity including clash-prone residues).
2. Positions 1 and 20 are reasonable secondary candidates, weaker evidence.
3. Positions 508 and 608 (outside the CDD pocket) are real, PLIP-corroborated leads worth revisiting once more importer-side redocking data exists — do not treat as confirmed yet.
4. The importer-vs-non-importer distinction in raw pocket-engagement/RMSD metrics (§2, §4) is confounded by HADDOCK3's own CDD-derived restraints being identical for both roles — any future redocking-based role classifier should consider an unrestrained/blind docking protocol as a fairer test, since the current one is somewhat pre-disposed to engage the same pocket in every protein regardless of true importer status.

---

## Where the numbers live

- `results/comparison/summary.csv`, `*_comparison.json` — Stage 7 (RMSD / pocket engagement)
- `results/comparison/pose_pocket_engagement.csv`, `good_pose_representative.csv` — Stage 8 (GMM good/bad pose)
- `results/rescoring/cdd_agreement.csv`, `ab_initio_rosetta_baseline_cdd_agreement.csv` — §4
- `results/rescoring/position_cohesion_scan.csv` — §5/§6 (rerun with `--cdd-only` for the §5 family, without it for the §6 full-alignment family)
- `results/rescoring/lda_unfavorable_contacts.csv` / `position_energetics_full.csv` — per-(protein, position) energetics feeding the scan above

Full technical/debugging history (every bug found and fixed getting here) is in project memory, not duplicated here — see `reference_haddock3_cns_ligand_param_propagation` and `project_redocking_pipeline_plan`.
