"""Repeat the two evaluation tasks on two public HTE yield datasets from other laboratories.

Datasets (not redistributed; download them from the rxn_yields repository of ref. 12 and pass the directory):
- Buchwald-Hartwig, Ahneman et al. (ref. 10): data/Buchwald-Hartwig/Dreher_and_Doyle_input_data.xlsx, sheet FullCV_01
  (3,955 reactions; one amine, so a substrate pair is an aryl halide).
- Suzuki-Miyaura, Perera et al. (ref. 2): data/Suzuki-Miyaura/aap9112_Data_File_S1.xlsx, Sheet1 (5,760 reactions;
  15 electrophile-nucleophile pairs). Structures follow the compound names in the original Supplementary Materials.

Features and model follow the main analysis: 1,024-bit Morgan fingerprints (radius 2) of substrate 1, substrate 2 and
product, and the 100-tree random forest. Because these two datasets cross their condition components factorially,
conditions are one-hot encoded per component (ligand, base, and additive or solvent) rather than as one identifier.
Held-out wells use shuffled five-fold KFold; new combinations withhold whole pairs (five folds of pairs); five split
seeds each. R2 is pooled against the dataset's full yield variance. Also reported: the between-pair share of yield
variance, the R2 of quoting each held-out well's pair mean from its training wells, and the pair-level and within-pair
R2 of the forest's predictions. Output: external_hte_summary.json in a separate directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

RDLogger.DisableLog("rdApp.*")
SEEDS = (42, 20261007, 20261008, 20261009, 20261010)
FILES = {
    "Buchwald-Hartwig (Ahneman et al.)": ("Dreher_and_Doyle_input_data.xlsx",
                                         "d7f318abfbb42eaadca482b1e9aa54540eeed60e0089b6a0eb4e75366c31ee2e"),
    "Suzuki-Miyaura (Perera et al.)": ("aap9112_Data_File_S1.xlsx",
                                       "a869e020ba31bd5676c67a4791c3b7384711b5216de6af444b8cd0a24c284640"),
}
INDAZOLE = "Cc1ccc2n(C3CCCCO3)ncc2c1{}"  # 5-methyl-1-(tetrahydro-2H-pyran-2-yl)-1H-indazol-4-yl
QUINOLINE = "{}c1ccc2ncccc2c1"            # substituent on quinolin-6-yl
SUZUKI_1 = {"1a, 6-Cl-Q": QUINOLINE.format("Cl"), "1b, 6-Br-Q": QUINOLINE.format("Br"),
            "1c, 6-OTf-Q": QUINOLINE.format("FC(F)(F)S(=O)(=O)O"), "1d, 6-I-Q": QUINOLINE.format("I"),
            "1e, 6-BOH2-Q": QUINOLINE.format("OB(O)"), "1f, 6-BPin-Q": QUINOLINE.format("CC1(C)OB(OC1(C)C)"),
            "1g, 6-BF3K-Q": QUINOLINE.format("F[B-](F)(F)")}
SUZUKI_2 = {"2a, Boronic Acid": INDAZOLE.format("B(O)O"), "2b, Boronic Ester": INDAZOLE.format("B1OC(C)(C)C(C)(C)O1"),
            "2c, Trifluoroborate": INDAZOLE.format("[B-](F)(F)F"), "2d, Bromide": INDAZOLE.format("Br")}
SUZUKI_PRODUCT = INDAZOLE.format("-c1ccc2ncccc2c1")  # 6-(indazol-4-yl)quinoline
TOLUIDINE = "Cc1ccc(N)cc1"
RF = dict(n_estimators=100, max_depth=None, max_features=1.0, min_samples_leaf=1, min_samples_split=2,
          bootstrap=True, random_state=42, n_jobs=-1)


def checked(directory: Path, name: str, sha: str) -> Path:
    path = directory / name
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise SystemExit(f"STOP: {name} is not the file this analysis was run on")
    return path


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Unparseable SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def amination_product(aryl_halide: str) -> str:
    rxn = AllChem.ReactionFromSmarts("[c:1][Cl,Br,I]>>[c:1]Nc1ccc(C)cc1")
    products = {Chem.MolToSmiles(p[0]) for p in rxn.RunReactants((Chem.MolFromSmiles(aryl_halide),))}
    if len(products) != 1:
        raise ValueError(f"Ambiguous product for {aryl_halide}")
    return products.pop()


def load(directory: Path) -> dict[str, pd.DataFrame]:
    name, sha = FILES["Buchwald-Hartwig (Ahneman et al.)"]
    bh = pd.read_excel(checked(directory, name, sha), sheet_name="FullCV_01")
    bh = pd.DataFrame({"sub_1": bh["Aryl halide"].map(canonical), "sub_2": canonical(TOLUIDINE),
                       "product": bh["Aryl halide"].map(lambda s: canonical(amination_product(s))),
                       "c1": bh.Ligand.astype(str), "c2": bh.Base.astype(str), "c3": bh.Additive.astype(str),
                       "y": bh.Output.astype(float) / 100})
    name, sha = FILES["Suzuki-Miyaura (Perera et al.)"]
    sm = pd.read_excel(checked(directory, name, sha), sheet_name="Sheet1")
    sm = pd.DataFrame({"sub_1": sm.Reactant_1_Short_Hand.map(SUZUKI_1).map(canonical),
                       "sub_2": sm.Reactant_2_Name.map(SUZUKI_2).map(canonical),
                       "product": canonical(SUZUKI_PRODUCT),
                       "c1": sm.Ligand_Short_Hand.astype(str), "c2": sm.Reagent_1_Short_Hand.astype(str),
                       "c3": sm.Solvent_1_Short_Hand.astype(str),
                       "y": sm.Product_Yield_PCT_Area_UV.astype(float) / 100})
    out = {"Buchwald-Hartwig (Ahneman et al.)": bh, "Suzuki-Miyaura (Perera et al.)": sm}
    for name, frame in out.items():
        if frame.isna().any().any():
            raise ValueError(f"{name}: unmapped structure or missing yield")
        frame["pair"] = frame.sub_1 + "." + frame.sub_2
        frame["condition"] = frame.c1 + "|" + frame.c2 + "|" + frame.c3
    return out


def fingerprint(smiles: str) -> np.ndarray:
    bits = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(smiles), 2, nBits=1024)
    arr = np.zeros((1024,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(bits, arr)
    return arr


def features(frame: pd.DataFrame) -> np.ndarray:
    cache = {s: fingerprint(s) for s in pd.unique(frame[["sub_1", "sub_2", "product"]].values.ravel())}
    conditions = pd.get_dummies(frame[["c1", "c2", "c3"]]).to_numpy(np.uint8)
    return np.hstack([np.stack(frame[c].map(cache).to_list()) for c in ("sub_1", "sub_2")] + [conditions]
                     + [np.stack(frame["product"].map(cache).to_list())]).astype(np.float32)


def pair_folds(pairs: pd.Series, seed: int) -> np.ndarray:
    unique = np.array(sorted(pairs.unique()))
    order = np.random.default_rng(seed).permutation(len(unique))
    fold_of = {unique[i]: k % 5 for k, i in enumerate(order)}
    return pairs.map(fold_of).to_numpy()


def decomposition(pairs: np.ndarray, y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    d = pd.DataFrame({"pair": pairs, "y": y, "p": p})
    yb = d.groupby("pair").y.transform("mean").to_numpy()
    pb = d.groupby("pair").p.transform("mean").to_numpy()
    return (float(1 - np.sum((yb - pb) ** 2) / np.sum((yb - y.mean()) ** 2)),
            float(1 - np.sum(((y - yb) - (p - pb)) ** 2) / np.sum((y - yb) ** 2)))


def analyse(frame: pd.DataFrame) -> dict:
    X, y, pairs = features(frame), frame.y.to_numpy(float), frame.pair.to_numpy()
    sst = float(np.sum((y - y.mean()) ** 2))
    pair_mean = frame.groupby("pair").y.transform("mean").to_numpy()
    share = float(np.sum((pair_mean - y.mean()) ** 2) / sst)
    out = {"reactions": len(frame), "pairs": int(frame.pair.nunique()), "conditions": int(frame.condition.nunique()),
           "mean_yield_percent": 100 * float(y.mean()), "pair_variance_share": share}
    per_seed = []
    for seed in SEEDS:
        record = {"seed": seed}
        for task in ("row", "group"):
            folds = (np.empty(len(y), dtype=int) if task == "row" else pair_folds(frame.pair, seed))
            if task == "row":
                for k, (_, test) in enumerate(KFold(5, shuffle=True, random_state=seed).split(X)):
                    folds[test] = k
            pred, sister = np.empty(len(y)), np.empty(len(y))
            for k in range(5):
                test, train = folds == k, folds != k
                pred[test] = RandomForestRegressor(**RF).fit(X[train], y[train]).predict(X[test])
                means = pd.Series(y[train]).groupby(pairs[train]).mean()
                sister[test] = pd.Series(pairs[test]).map(means).fillna(y[train].mean()).to_numpy()
            level, within = decomposition(pairs, y, pred)
            record[f"{task}_r2"] = float(1 - np.sum((y - pred) ** 2) / sst)
            record[f"{task}_pair_level_r2"], record[f"{task}_within_pair_r2"] = level, within
            if task == "row":
                record["row_sister_well_mean_r2"] = float(1 - np.sum((y - sister) ** 2) / sst)
            else:
                seen = np.zeros(len(y), dtype=bool)
                for k in range(5):
                    test, train = folds == k, folds != k
                    seen[test] = (frame.sub_1[test].isin(set(frame.sub_1[train])).to_numpy()
                                  & frame.sub_2[test].isin(set(frame.sub_2[train])).to_numpy())
                record["group_both_seen_share"] = float(seen.mean())
        per_seed.append(record)
        print(json.dumps(record), flush=True)
    table = pd.DataFrame(per_seed)
    out.update({k: float(table[k].mean()) for k in table.columns if k != "seed"})
    out["r2_lost"] = out["row_r2"] - out["group_r2"]
    out["per_seed"] = per_seed
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="directory holding the two downloaded files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = {name: analyse(frame) for name, frame in load(args.data_dir).items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "external_hte_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({n: {k: v for k, v in s.items() if k != "per_seed"} for n, s in summary.items()}, indent=2))


if __name__ == "__main__":
    sys.exit(main())
