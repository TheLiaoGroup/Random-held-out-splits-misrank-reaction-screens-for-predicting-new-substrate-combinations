# Held-out wells and new substrate combinations: code and data

This directory contains the data, saved predictions and code for **Random held-out splits misrank reaction screens for predicting new substrate combinations**: 340,178 numerical yields from 18 high-throughput screens in 17 reaction families. It runs on its own; it is deposited at Zenodo (https://doi.org/10.5281/zenodo.23319287) and maintained on GitHub (https://github.com/TheLiaoGroup/Held-out-wells-misrank-reaction-screens-for-predicting-new-substrate-combinations). File names keep the project prefix `p4_`. The code is released under the MIT License and the data and results under CC BY 4.0 (see `LICENSE`).

## Data

| File | Content |
|---|---|
| `data/p4_records.csv.gz` | One row per numerical yield (340,178 rows): `screen`, `row_uid`, `plate_group`, `component_group` (connected plate/pair group used for new-combination folds), `pair_key`, `condition_id`, `sub_1_smiles`, `sub_2_smiles`, `product_smiles`, `yield_fraction` (calibrated assay yield, 0–1; explicit zeros are wells without detected product). Rows of each screen are in the order used for held-out-well folds. |
| `data/p4_grouped_folds.csv.gz` | The frozen new-combination fold (0–4) of every record for each of the five split seeds. |
| `data/p4_screens.json` | Per-screen metadata: reaction family, record count, mean yield, total sum of squares, the ordered condition vocabulary used for one-hot features, and the pooled scores of the original screen-specific models (random forests for Suzuki and Suzuki Bpin, XGBoost otherwise). |
| `data/benchmarks/` | The 288 within-plate repeated-reaction yields used for Extended Data Fig. 1, with their reaction definitions. |

Conditions are identified by their recorded condition identifiers; every model uses them as one-hot features.

## Reproduce the manuscript results

Run from this directory using **Python 3.13.3** and the versions in `requirements-p4.txt`. All scripts check the three data files against their SHA-256 in `RELEASE_MANIFEST.json`. The released results are in `output/`; the figures are drawn from them directly:

```powershell
python -m pip install -r requirements-p4.txt
python p4_manuscript_figures_20261008.py --output-dir ../recomputed/figures
python p4_rf_figure.py --diagnostics-dir output/p4_unified_rf/diagnostics --output-dir ../recomputed/figures
```

`p4_manuscript_figures_20261008.py` draws Figures 1–3 and 5 and Extended Data Figures 1–3; `p4_rf_figure.py` draws Figure 4. Both take their fonts, text sizes and panel-label placement from `p4_figure_style.py`; the figures use Arial, and where Arial is not installed matplotlib substitutes another sans-serif font.

To recompute the results, first fit the models on the released records, condition vocabularies and folds, then run the diagnostics on their out-of-fold predictions. Add `--screen Minisci` to a fitting command to fit one screen.

```powershell
python p4_rf_oof.py --package-dir . --output-dir ../rf --threads 4 --fold-workers 2
python p4_rf_oof_diagnostics.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_baseline_weighting_diagnostics.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_ranking_structure_diagnostics.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_additive_baseline.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_additive_least_squares.py --package-dir . --results-dir ../rf --output-dir ../rf/diagnostics
python p4_mlp_oof.py --package-dir . --output-dir ../mlp --workers 4
python score_predictions.py score --predictions ../mlp/mlp_predictions_all.csv.gz --model-name MLP --output-dir ../mlp/scores
python p4_mlp_comparison.py --package-dir . --predictions ../mlp/mlp_predictions_all.csv.gz --output-dir ../mlp/comparison
```

- `p4_rf_oof.py` fits the common 100-tree random forest (900 fits) and writes per-well out-of-fold predictions with columns `screen,row_uid,protocol,seed,fold,y_true,y_pred,n_unseen,training_mean`; `n_unseen` counts the substrate roles whose recorded SMILES was absent from that fold's training records.
- `p4_rf_oof_diagnostics.py`: pooled metrics, rankings, top-k selection curves, screen omission, substrate exposure and matched cohorts.
- `p4_baseline_weighting_diagnostics.py`: the training-fold median reference, pair-equal weighting and the both-seen cohort.
- `p4_ranking_structure_diagnostics.py`: between-pair and between-condition variance shares, the between-pair share after condition effects, pairing sparsity, yield distributions, the sister-well reference, the pair-level decomposition and pairwise order reversals.
- `p4_additive_baseline.py` and `p4_additive_least_squares.py`: the additive substrate-1 + substrate-2 + condition identity model, fitted by backfitting and by minimum-norm least squares in every fold.
- `p4_mlp_oof.py` fits the neural network (900 fits, about five hours on 11 processes); `p4_mlp_comparison.py` compares it with the forest.
- The pretrained reaction representation: copy `output/p4_rxnfp/embeddings.csv.gz` and `reactions.csv` to a working directory and run `python p4_rxnfp_oof.py fit --output-dir <directory>`; then score and compare its predictions as for the network, with `--label rxnfp`. The embeddings were computed by the `embed` stage in a separate Python 3.9 environment with rxnfp 0.1.0, transformers 4.30.2 and PyTorch 2.2.2.

In our checks, refits of Minisci and Grignard reproduced the original forest predictions to within 5 × 10⁻¹⁶ (the order of multithreaded tree averaging), and a refit of Minisci reproduced the original network predictions exactly.

## Released results

- `output/p4_unified_rf/diagnostics/`: every diagnostic of the random forest reported in the manuscript; `output/p4_unified_rf/run_manifest.json` records the model and its fitted predictions.
- `output/p4_mlp/` and `output/p4_rxnfp/`: scores and comparisons of the neural network and the pretrained reaction representation; `output/p4_rxnfp/` also holds the reaction SMILES per substrate pair and their rxnfp 0.1.0 embeddings.
- `output/p4_external/external_hte_summary.json`: both tasks on the public Buchwald–Hartwig (Ahneman et al.) and Suzuki–Miyaura (Perera et al.) datasets. The check needs `Dreher_and_Doyle_input_data.xlsx` and `aap9112_Data_File_S1.xlsx` from github.com/rxn4chemistry/rxn_yields: `python p4_external_hte_check.py --data-dir <folder> --output-dir ../external` (the script checks their SHA-256).
- `output/p4_original_models/model_parameters.json`: estimators and parameters of the original screen-specific models, whose pooled scores are in `data/p4_screens.json`.

## Score another model

```powershell
python score_predictions.py template --output ..\model_predictions.csv.gz
python score_predictions.py score --predictions ..\model_predictions.csv.gz --model-name "My model" --output-dir ..\other-model-results
```

The template lists every record with its fold under both protocols and all five seeds (3,401,780 rows); fill `y_pred` with a yield fraction predicted by a model trained on the other four folds of the same screen, protocol and seed. Add `--screen`, `--protocol` or `--seed` to the template command for a smaller task. Scoring requires the complete file, rejects unknown, duplicate or missing records and wrong folds, and writes pooled R², MAE and RMSE, rankings, selection curves and omission diagnostics.

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
