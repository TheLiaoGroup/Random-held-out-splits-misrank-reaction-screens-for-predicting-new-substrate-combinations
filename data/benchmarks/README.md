# Representative within-plate repeatability

`platform_validation_all_wells.csv` provides all 288 recorded integer-percentage yields: one 96-well plate per reaction class, repeating a single high-yielding reaction through setup, reaction execution, workup, analysis and data processing. The platform used individual glass reaction tubes and magnetic stir bars within screw-sealed aluminium blocks. Each well contained 0.025 mmol of the limiting substrate.

| Reference | Limiting substrate | Partner | Reagents | Solvent | Temperature and time |
|---|---|---|---|---|---|
| Acylation | 4-aminoacetophenone | 2-(4-methylphenyl)acetic acid, 1.2 equivalents | HATU, 1.5 equivalents; DIPEA, 3.0 equivalents | DMF | 30 °C, 3 h |
| Suzuki-Miyaura | 3-bromo-2-methylpyridine | 4-methoxyphenylboronic acid, 1.5 equivalents | XPhos Pd G3, 5 mol%; K3PO4, 3 equivalents | 1,4-dioxane/water, 4:1 | 100 °C, 12 h |
| Buchwald-Hartwig | 1-bromo-3,5-dimethoxybenzene | 4-methoxyaniline, 1.2 equivalents | Pd2(dba)3, 10 mol%; XantPhos, 20 mol%; K3PO4, 3 equivalents | 1,4-dioxane | 100 °C, 16 h |

The general platform readout screened product formation by DESI-TOF-MS, used selected UPLC-UV/MS measurements for relative product responses, and selected 1H NMR measurements for absolute-yield anchoring and response calibration. The recorded benchmark tables do not assign a separate analytical tier to every well. Reaction solvent volumes and concentrations are not given for these three plates; the 900-microlitre glass-tube specification is vessel capacity, not reaction volume. Representative dispensing or dilution examples elsewhere in the source are not assigned to these plates.

Calculate sample SD with `ddof=1` over each plate's 96 yields. The means are 68.78125%, 79.9479167% and 75.56250%, and the SDs are 3.9528887, 4.6438863 and 5.2008350 yield percentage points, respectively. Extended Data Figure 1 compares these SDs with prediction RMSE for different experimental populations; the ratios are descriptive error-scale comparisons.

## Provenance

The values and reaction definitions come from the laboratory's **Data-centric ultra-high-throughput experimentation enables AI-ready reaction-space mapping**, corrected 0612 Supplementary Information: Section S3, subsection 4, and reaction schemes accompanying Tables S8-S10. General equipment and readout descriptions are in Sections S2.2, S2.4, S4.4 and S4.7. Original DOCX SHA-256: `ce82f874b8682bf54bf171d90ee83caa663c629409a96e307978dec4a3ab0f5b`.

That source is a retained laboratory manuscript record; no public DOI is asserted. The reaction definitions above and all well-level benchmark values are included here, and the current P4 Methods gives the same reaction conditions, plate design and statistical definition. `platform_validation_metadata.json` preserves the original source hashes and historical audit provenance; its source paths are reduced to file names and are not runtime dependencies.
