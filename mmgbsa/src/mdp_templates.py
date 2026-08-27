"""
mmgbsa/src/mdp_templates.py
===========================
GROMACS .mdp templates for the Stage 2/3 protocol. Amber99SB-ILDN + TIP3P,
PME, LINCS on h-bonds, 2 fs. Kept here as plain strings so prep_systems.py
stays readable; `{seed}` in PROD is filled per replica by run_md_batch.py's
grompp step (or here with a fixed seed for the smoke test).

Protocol (see mmgbsa/README.md):
  EM   : steepest descent to Fmax < 1000 kJ/mol/nm
  NVT  : 100 ps, 300 K (V-rescale), protein heavy atoms + ligand restrained (POSRES + POSRES_LIG)
  NPT  : 900 ps, 1 bar (C-rescale), same restraints  -> ~1 ns equilibration total
  PROD : 5 ns, 1 bar (Parrinello-Rahman), ONLY C-alpha restrained (POSRES_CA, 100 kJ/mol/nm2)
         -- no membrane in this first pass, so the weak CA tether stops the
         transporter fold drifting while leaving the pocket + ligand free.
"""
from __future__ import annotations

IONS = """\
; ion placement only
integrator      = steep
emtol           = 1000.0
nsteps          = 5000
nstlist         = 10
cutoff-scheme   = Verlet
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""

EM = """\
integrator      = steep
emtol           = 1000.0
emstep          = 0.01
nsteps          = 50000
nstlist         = 10
cutoff-scheme   = Verlet
ns_type         = grid
coulombtype     = PME
rcoulomb        = 1.0
rvdw            = 1.0
pbc             = xyz
"""

_COMMON_MD = """\
cutoff-scheme            = Verlet
ns_type                 = grid
nstlist                 = 20
rcoulomb                = 1.0
rvdw                    = 1.0
coulombtype             = PME
pme_order               = 4
fourierspacing          = 0.12
constraints             = h-bonds
constraint_algorithm    = lincs
lincs_iter              = 1
lincs_order             = 4
pbc                     = xyz
DispCorr                = EnerPres
"""

NVT = """\
define                  = -DPOSRES -DPOSRES_LIG
integrator              = md
nsteps                  = 50000      ; 100 ps @ 2 fs
dt                      = 0.002
nstxout-compressed      = 5000
nstenergy               = 5000
nstlog                  = 5000
continuation            = no
gen_vel                 = yes
gen_temp                = 300
gen_seed                = -1
""" + _COMMON_MD + """\
tcoupl                  = V-rescale
tc-grps                 = Protein_GA1 Water_and_ions
tau_t                   = 0.1   0.1
ref_t                   = 300   300
pcoupl                  = no
"""

NPT = """\
define                  = -DPOSRES -DPOSRES_LIG
integrator              = md
nsteps                  = 450000     ; 900 ps @ 2 fs
dt                      = 0.002
nstxout-compressed      = 5000
nstenergy               = 5000
nstlog                  = 5000
continuation            = yes
gen_vel                 = no
""" + _COMMON_MD + """\
tcoupl                  = V-rescale
tc-grps                 = Protein_GA1 Water_and_ions
tau_t                   = 0.1   0.1
ref_t                   = 300   300
pcoupl                  = C-rescale
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoord_scaling        = com
"""

# {seed} filled per replica (run_md_batch.py). 5 ns, frame every 10 ps -> 500 frames.
PROD = """\
define                  = -DPOSRES_CA
integrator              = md
nsteps                  = 2500000    ; 5 ns @ 2 fs
dt                      = 0.002
nstxout-compressed      = 5000       ; 10 ps
compressed-x-grps       = Protein_GA1
nstenergy               = 5000
nstlog                  = 50000
continuation            = yes
gen_vel                 = no
ld_seed                 = {seed}
""" + _COMMON_MD + """\
tcoupl                  = V-rescale
tc-grps                 = Protein_GA1 Water_and_ions
tau_t                   = 0.1   0.1
ref_t                   = 300   300
pcoupl                  = Parrinello-Rahman
pcoupltype              = isotropic
tau_p                   = 2.0
ref_p                   = 1.0
compressibility         = 4.5e-5
refcoord_scaling        = com
"""


def prod_mdp(seed: int, nsteps: int | None = None) -> str:
    text = PROD.format(seed=seed)
    if nsteps is not None:  # smoke test shortens this
        text = text.replace("nsteps                  = 2500000    ; 5 ns @ 2 fs",
                            f"nsteps                  = {nsteps}")
    return text
