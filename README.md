# Whether a reaction screen predicts new substrate combinations can be anticipated from its yield table: code and data

This directory contains the released records, results and code for **Whether a reaction screen predicts new substrate combinations can be anticipated from its yield table**. The 19 high-throughput screens (arrays) in 17 transformations contain 406,909 records. The analysis uses 378,070 records: 357,356 recorded numerical yields and 20,714 zeros assigned by the measurement-status rule below. The package contains the data and summaries needed to redraw the figures. The repository is https://github.com/TheLiaoGroup/Yield-table-predictability. The DOI https://doi.org/10.5281/zenodo.23319287 is reserved for an unpublished Zenodo draft and does not yet identify a published dataset. File names keep the project prefix `p4_`. The code is released under the MIT License and the data and results under CC BY 4.0 (see `LICENSE`).

## Data

| File | Content |
|---|---|
| `data/p4_records.csv.gz` | All 406,909 records: `screen`, `row_uid`, `measurement_status`, `plate_group`, `component_group` (connected plate/pair group used for new-combination folds), `pair_key`, `condition_id`, `sub_1_smiles`, `sub_2_smiles`, `product_smiles`, `yield_fraction`, `analysis_eligible`. Records flagged eligible with status `quantified` or `zero_by_rule` enter prediction tasks. Their order within each screen defines the held-out-well folds. |
| `data/p4_grouped_folds.csv.gz` | The frozen new-combination fold (0–4) of every analyzed record for each of the five split seeds (1,890,350 rows). The original all-record plate/pair groups and fold assignments are retained after exclusion and eligibility filtering. |
| `data/p4_screens.json` | Per-screen metadata under `sources`: `reaction_family` (the screen's named transformation, 17 values for 19 screens; Suzuki and Suzuki Bpin share one, as do the two Buchwald–Hartwig screens), measurement-status counts, analyzed record count, mean yield, total sum of squares and the ordered condition vocabulary used for one-hot features. Model scores are stored separately under `output/`. |
| `data/benchmarks/` | The 288 within-plate repeated-reaction yields used for Supplementary Fig. S1, with their reaction definitions. |

The benchmark README's historical Extended Data Figure 1 label corresponds to
the current Supplementary Figure S1.

The four measurement statuses are `quantified` (357,356), `zero_by_rule` (20,714), `detected_not_quantified` (24,226) and `no_screen_record` (4,613). A finite source yield takes priority and is `quantified`. For a blank source yield, `yield_clf=0` assigns `zero_by_rule` and a numerical yield of zero; `yield_clf=1` assigns `detected_not_quantified`; any other or missing classification assigns `no_screen_record`. The last two statuses retain a missing numerical yield and do not enter prediction or scoring. Conditions are identified by their recorded condition identifiers; every model uses them as one-hot features.

## Reproduce the manuscript results

Run from this directory using **Python 3.13.3** and the versions in `requirements-p4.txt`. The fitting and diagnostic loaders check the frozen inputs against `RELEASE_MANIFEST.json`; the figure scripts read released summaries and check their required provenance links. The released results are in `output/`; the figures are drawn from them directly:

```powershell
python -m pip install -r requirements-p4.txt
python p4_chem_figures.py --output-dir figures
python p4_supplementary_figures.py --output-dir figures
```

`p4_chem_figures.py` draws Figures 1–5 and the graphical abstract (TOC);
`p4_supplementary_figures.py` draws the provisionally numbered Supplementary
Figures S1–S7. `chemical_assets/` contains ten editable ChemDraw CDXML masters
and their native SVG exports: eight carbonyl partners and the Grignard-addition
and esterification schemes. `assets.json` records their identities, native-export
version, hashes and physical dimensions. `p4_structure_drawing.py` places these
vectors at their original size (14.4 pt bonds, 0.6 pt lines and 8 pt Arial), without
individually shrinking structures. ChemDraw is needed only to edit and re-export
the chemical masters; redrawing the figures uses the supplied SVG files. Both renderers use
`p4_figure_style.py`, with Arial at 9 and 8 pt. Install genuine Arial font files;
font substitution is not supported. Figure 1's layered SVG is the editable
master; its PDF and PNG are converted by the main script using `svglib`,
`reportlab` and `PyMuPDF`. The conversion locates `arial.ttf`, `arialbd.ttf` and
`ariali.ttf` in the system font directory. All figure values are available in the
released data and result files; Chem delivery does not include a separate Source
Data workbook.

Per-well model predictions and resumable fitting checkpoints are not included in this package. The figures can be regenerated from the released summaries without fitting. Recomputing prediction-level diagnostics requires the corresponding saved predictions or refitting the models. All result directories below should be outside any cloud-synchronized directory. Add `--screen Minisci` to the RF, original-model or MLP fitting command to fit one screen.

Fit and score the original screen-specific sensitivity models before the unified-RF diagnostics:

```powershell
python p4_original_models.py fit --package-dir . --output-dir ../original-models --threads 4
python p4_original_models.py score --package-dir . --output-dir ../original-models
python p4_rf_oof.py --package-dir . --output-dir ../rf --threads 4 --fold-workers 2
python p4_rf_oof_diagnostics.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics --original-results ../original-models/original_pooled_model_results.json
python p4_baseline_weighting_diagnostics.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_ranking_structure_diagnostics.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_additive_baseline.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_additive_least_squares.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
```

For the comparison scripts, first copy the completed RF diagnostics into `output/p4_unified_rf/diagnostics/` in a separate working copy of this package. The comparison scripts read those released diagnostic paths. Keep the deposited package unchanged.

```powershell
python p4_mlp_oof.py --package-dir . --output-dir ../mlp --workers 4
python score_predictions.py score --predictions ../mlp/mlp_predictions_all.csv.gz --model-name MLP --output-dir ../mlp/scores
python p4_mlp_comparison.py --package-dir . --predictions ../mlp/mlp_predictions_all.csv.gz --label mlp --description "MLPRegressor (256, 128)" --output-dir ../mlp/comparison
```

- `p4_original_models.py` fits the original per-screen estimator specifications: random forests for Suzuki and Suzuki Bpin, XGBoost for the other screens. The added Buchwald-Hartwig CMC screen uses the Buchwald-Hartwig specification. Its scoring stage writes `original_pooled_model_results.json` for the sensitivity comparison.
- `p4_rf_oof.py` fits the common 100-tree random forest (950 fits) and writes per-well out-of-fold predictions with columns `screen,row_uid,protocol,seed,fold,y_true,y_pred,n_unseen,training_mean`; `n_unseen` counts the substrate roles whose recorded SMILES was absent from that fold's training records.
- `p4_rf_oof_diagnostics.py`: pooled metrics, rankings, top-k selection curves, screen omission, substrate exposure and matched cohorts.
- `p4_baseline_weighting_diagnostics.py`: the training-fold median reference, pair-equal weighting and the both-seen cohort.
- `p4_ranking_structure_diagnostics.py`: between-pair and between-condition variance shares, the between-pair share after condition effects, pairing sparsity, yield distributions, the sister-well reference, the pair-level decomposition and pairwise order reversals.
- `p4_additive_baseline.py` and `p4_additive_least_squares.py`: the additive substrate-1 + substrate-2 + condition identity model, fitted by backfitting and by minimum-norm least squares in every fold.
- `p4_mlp_oof.py` fits the neural network (950 fits); `p4_mlp_comparison.py` compares it with the forest. Runtime depends on the execution environment.
- The pretrained reaction representation: copy `output/p4_rxnfp/embeddings.csv.gz` and `reactions.csv` to a working directory and run `python p4_rxnfp_oof.py fit --output-dir <directory>`; then score and compare its predictions as for the network, with `--label rxnfp`. The embeddings were computed by the `embed` stage in a separate Python 3.9 environment with rxnfp 0.1.0, transformers 4.30.2 and PyTorch 2.2.2.

For a rerun without refitting the primary models, reuse the completed `../rf`, `../original-models`, `../mlp` and reaction-representation run directories: omit the primary-model fitting commands and run the scoring, diagnostic and comparison commands above. The two additive-reference scripts refit their reference models. The RF run directory must include its manifest, frozen configuration and all per-screen out-of-fold prediction files.

## Released results

- `output/p4_unified_rf/diagnostics/`: every diagnostic of the random forest reported in the manuscript; `output/p4_unified_rf/run_manifest.json` records the model and the metadata and hashes of fitted predictions stored outside this package.
- `output/p4_mlp/` and `output/p4_rxnfp/`: scores and comparisons of the neural network and the pretrained reaction representation; `output/p4_rxnfp/` also holds the reaction SMILES per substrate pair and their rxnfp 0.1.0 embeddings.
- `output/p4_external/external_hte_summary.json`: both tasks on the public Buchwald–Hartwig (Ahneman et al.) and Suzuki–Miyaura (Perera et al.) datasets. The check needs `Dreher_and_Doyle_input_data.xlsx` and `aap9112_Data_File_S1.xlsx` from github.com/rxn4chemistry/rxn_yields: `python p4_external_hte_check.py --data-dir <folder> --output-dir ../external` (the script checks their SHA-256).
- `output/p4_original_models/model_parameters.json`: estimators and parameters of the original screen-specific models. Their recomputed pooled scores are in `output/p4_original_models/original_pooled_model_results.json`.

## Score another model

```powershell
python score_predictions.py template --output ..\model_predictions.csv.gz
python score_predictions.py score --predictions ..\model_predictions.csv.gz --model-name "My model" --output-dir ..\other-model-results
```

The template lists every analyzed record with its fold under both protocols and all five seeds (3,780,700 rows); fill `y_pred` with a yield fraction predicted by a model trained on the other four folds of the same screen, protocol and seed. Add `--screen`, `--protocol` or `--seed` to the template command for a smaller task. Scoring requires the complete file, rejects unknown, duplicate or missing records and wrong folds, and writes pooled R², MAE and RMSE, rankings, selection curves and omission diagnostics.

| Column | Definition |
|---|---|
| `screen` | Screen name. |
| `row_uid` | Record identifier within that screen. |
| `protocol` | `row` for held-out wells, `group` for new combinations. |
| `seed` | `42`, `20261007`, `20261008`, `20261009` or `20261010`. |
| `fold` | Test fold, `0`–`4`. |
| `y_pred` | Predicted yield fraction; values outside 0–1 are scored without clipping. |

Held-out-well folds are shuffled five-fold KFold splits of each screen's records in file order; new-combination folds are read from `data/p4_grouped_folds.csv.gz`. Every seed pools its five test folds before computing R² against the screen's full yield variance.

`RELEASE_MANIFEST.json` lists every file with its size and SHA-256.
