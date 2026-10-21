# Representative within-plate repeatability

`platform_validation_all_wells.csv` provides all 288 recorded integer-percentage yields: one 96-well plate per reaction class, repeating a single high-yielding reaction through setup, reaction execution, workup, analysis and data processing. The platform used individual glass reaction tubes and magnetic stir bars within screw-sealed aluminium blocks. Each well contained 0.025 mmol of the limiting substrate.

| Reference | Limiting substrate | Partner | Reagents | Solvent | Temperature and time |
|---|---|---|---|---|---|
| Acylation | 4-aminoacetophenone | 2-(4-methylphenyl)acetic acid, 1.2 equivalents | HATU, 1.5 equivalents; DIPEA, 3.0 equivalents | DMF | 30 C, 3 h |
| Suzuki-Miyaura | 3-bromo-2-methylpyridine | 4-methoxyphenylboronic acid, 1.5 equivalents | XPhos Pd G3, 5 mol%; K3PO4, 3 equivalents | 1,4-dioxane/water, 4:1 | 100 C, 12 h |
| Buchwald-Hartwig | 1-bromo-3,5-dimethoxybenzene | 4-methoxyaniline, 1.2 equivalents | Pd2(dba)3, 10 mol%; XantPhos, 20 mol%; K3PO4, 3 equivalents | 1,4-dioxane | 100 C, 16 h |

The platform readout screened product formation by DESI-TOF-MS, used selected UPLC-UV/MS measurements for relative product responses, and selected 1H NMR measurements for absolute-yield anchoring and response calibration. The benchmark data do not assign a separate analytical tier to every well. Reaction solvent volumes and concentrations are not specified for these three plates; the 900-microlitre glass-tube specification is vessel capacity, not reaction volume. Plate identifiers, dates and independent day/batch replicate counts are not supplied with these benchmark data.

Calculate sample SD with `ddof=1` over each plate's 96 yields. The means are 68.78125%, 79.9479167% and 75.56250%, and the SDs are 3.9528887, 4.6438863 and 5.2008350 yield percentage points, respectively. Figure S1 compares these SDs with prediction RMSE for different experimental populations; the ratios are descriptive error-scale comparisons.

`platform_validation_metadata.json` contains the experimental definitions and statistics recomputed from these 288 yields. All recorded integer-percentage values are supplied; the original unrounded measurements are not included. The repeated high-yielding reactions provide representative end-to-end repeatability benchmarks rather than estimates of low-yield analytical precision for all substrates.
