"""Describe rank and selection agreement using immutable, saved P4 scores."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
from statistics import mean

from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "output" / "p4_all_reactions_20261008"
INPUT = RESULTS / "pooled_model_results.json"
OUTPUT = RESULTS / "saved_score_diagnostics.json"
METRICS = ("r2", "mae", "rmse")
PROTOCOLS = ("row", "group")
MEAN_YIELD_THRESHOLD = 0.05


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def descriptive_range(values: list[float | int]) -> dict:
    return {"min": min(values), "max": max(values), "range": max(values) - min(values)}


def validate_saved_scores(payload: dict) -> tuple[list[dict], list[int]]:
    if payload["status"] != "completed":
        raise ValueError("the saved pooled analysis is not complete")
    sources = payload["sources"]
    seeds = payload["protocol"]["seeds"]
    names = [source["source_name"] for source in sources]
    if len(sources) != 18 or len(set(names)) != len(names):
        raise ValueError("expected 18 distinct saved screens")
    if len(seeds) != 5 or len(set(seeds)) != len(seeds):
        raise ValueError("expected five distinct saved seeds")
    for source in sources:
        if not math.isfinite(source["mean_yield_fraction"]):
            raise ValueError(f"non-finite mean yield: {source['source_name']}")
        for protocol in PROTOCOLS:
            records = source["pooled_by_seed"][protocol]
            record_seeds = [record["seed"] for record in records]
            if len(record_seeds) != len(seeds) or set(record_seeds) != set(seeds):
                raise ValueError(f"missing or duplicate seeds: {source['source_name']} {protocol}")
            for metric in METRICS:
                values = [record[metric] for record in records]
                saved_mean = source["summary"][protocol][metric]["mean"]
                if any(value is None or not math.isfinite(value) for value in values):
                    raise ValueError(f"undefined saved scores: {source['source_name']} {protocol} {metric}")
                if not math.isfinite(saved_mean) or not math.isclose(
                    saved_mean, mean(values), rel_tol=1e-12, abs_tol=1e-14
                ):
                    raise ValueError(f"saved mean disagrees with seed scores: {source['source_name']} {metric}")
    return sources, seeds


def saved_score(source: dict, protocol: str, metric: str, seed: int | None) -> float:
    if seed is None:
        return source["summary"][protocol][metric]["mean"]
    return next(record[metric] for record in source["pooled_by_seed"][protocol] if record["seed"] == seed)


def metric_rho(sources: list[dict], metric: str, seed: int | None = None) -> float:
    direction = 1 if metric == "r2" else -1
    row = [direction * saved_score(source, "row", metric, seed) for source in sources]
    group = [direction * saved_score(source, "group", metric, seed) for source in sources]
    rho = float(spearmanr(row, group).statistic)
    if not math.isfinite(rho):
        raise ValueError(f"undefined Spearman correlation: {metric}, seed={seed}")
    return rho


def metric_concordance(sources: list[dict], seeds: list[int]) -> dict:
    by_seed = [
        {"seed": seed, "rho": {metric: metric_rho(sources, metric, seed) for metric in METRICS}}
        for seed in seeds
    ]
    return {
        "screen_count": len(sources),
        "screen_mean": {metric: metric_rho(sources, metric) for metric in METRICS},
        "by_seed": by_seed,
        "seed_rho_ranges": {
            metric: descriptive_range([record["rho"][metric] for record in by_seed])
            for metric in METRICS
        },
    }


def selection_curve(sources: list[dict], seed: int | None = None) -> dict:
    ordered = {
        protocol: sorted(sources, key=lambda source: (-saved_score(source, protocol, "r2", seed),
                                                       source["source_name"]))
        for protocol in PROTOCOLS
    }
    by_k = []
    for k in range(1, len(sources) + 1):
        selected = {protocol: ordered[protocol][:k] for protocol in PROTOCOLS}
        names = {protocol: [source["source_name"] for source in selected[protocol]] for protocol in PROTOCOLS}
        overlap = sorted(set(names["row"]) & set(names["group"]))
        group_r2_means = {
            protocol: mean([saved_score(source, "group", "r2", seed) for source in selected[protocol]])
            for protocol in PROTOCOLS
        }
        by_k.append({
            "k": k,
            "row_selected": names["row"],
            "group_selected": names["group"],
            "overlap": overlap,
            "overlap_count": len(overlap),
            "group_r2_mean_row_selected": group_r2_means["row"],
            "group_r2_mean_group_selected": group_r2_means["group"],
            "group_minus_row_selected_r2": group_r2_means["group"] - group_r2_means["row"],
        })
    return {
        "row_order": [source["source_name"] for source in ordered["row"]],
        "group_order": [source["source_name"] for source in ordered["group"]],
        "by_k": by_k,
    }


def retrospective_selection(sources: list[dict], seeds: list[int]) -> dict:
    by_seed = [{"seed": seed, **selection_curve(sources, seed)} for seed in seeds]
    statistics = ("overlap_count", "group_r2_mean_row_selected", "group_r2_mean_group_selected",
                  "group_minus_row_selected_r2")
    return {
        "screen_mean": selection_curve(sources),
        "by_seed": by_seed,
        "seed_ranges_by_k": [
            {"k": k, **{
                statistic: descriptive_range([record["by_k"][k - 1][statistic] for record in by_seed])
                for statistic in statistics
            }}
            for k in range(1, len(sources) + 1)
        ],
    }


def discordant_pairs(sources: list[dict], seed: int | None = None) -> dict:
    pairs = []
    tied_pair_count = 0
    for first, second in itertools.combinations(sorted(sources, key=lambda source: source["source_name"]), 2):
        row_difference = saved_score(first, "row", "r2", seed) - saved_score(second, "row", "r2", seed)
        group_difference = saved_score(first, "group", "r2", seed) - saved_score(second, "group", "r2", seed)
        if row_difference == 0 or group_difference == 0:
            tied_pair_count += 1
        elif (row_difference > 0) != (group_difference > 0):
            row_preferred, group_preferred = (first, second) if row_difference > 0 else (second, first)
            pairs.append({"row_preferred": row_preferred["source_name"],
                          "group_preferred": group_preferred["source_name"]})
    total_pairs = len(sources) * (len(sources) - 1) // 2
    return {
        "total_pairs": total_pairs,
        "tied_pair_count": tied_pair_count,
        "discordant_count": len(pairs),
        "discordant_fraction_all_pairs": len(pairs) / total_pairs,
        "pairs": pairs,
    }


def r2_discordance(sources: list[dict], seeds: list[int]) -> dict:
    by_seed = [{"seed": seed, **discordant_pairs(sources, seed)} for seed in seeds]
    return {
        "screen_mean": discordant_pairs(sources),
        "by_seed": by_seed,
        "seed_discordant_count_range": descriptive_range([record["discordant_count"] for record in by_seed]),
    }


def leave_one_out(sources: list[dict], seed: int | None = None) -> dict:
    records = [{
        "omitted_screen": omitted["source_name"],
        "rho": metric_rho([source for source in sources if source is not omitted], "r2", seed),
    } for omitted in sources]
    return {"remaining_screen_count": len(sources) - 1,
            "by_omitted_screen": records,
            "rho_range": descriptive_range([record["rho"] for record in records])}


def leave_one_screen_out_r2(sources: list[dict], seeds: list[int]) -> dict:
    by_seed = [{"seed": seed, **leave_one_out(sources, seed)} for seed in seeds]
    return {
        "screen_mean": leave_one_out(sources),
        "by_seed": by_seed,
        "seed_rho_ranges_by_omitted_screen": [{
            "omitted_screen": source["source_name"],
            **descriptive_range([record["by_omitted_screen"][index]["rho"] for record in by_seed]),
        } for index, source in enumerate(sources)],
    }


def build_diagnostics(payload: dict, input_path: Path, input_sha256: str) -> dict:
    sources, seeds = validate_saved_scores(payload)
    retained = [source for source in sources if source["mean_yield_fraction"] >= MEAN_YIELD_THRESHOLD]
    if len(retained) != 11:
        raise ValueError("expected 11 saved screens with mean yield >= 0.05")
    return {
        "schema_version": 1,
        "provenance": {
            "input_path": input_path.resolve().as_posix(),
            "input_sha256": input_sha256,
            "script_path": Path(__file__).resolve().as_posix(),
            "script_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "pooled_evaluation_script_path": (ROOT / "p4_pooled_evaluation_20261008.py").as_posix(),
            "pooled_evaluation_script_sha256_at_input_creation": payload["protocol"]["script_sha256"],
            "upstream_input_sha256": payload["input_sha256"],
            "screen_mean_score_field": "sources[].summary.{row,group}.{r2,mae,rmse}.mean",
            "seed_score_field": "sources[].pooled_by_seed.{row,group}[] matched on seed",
        },
        "definitions": {
            "score_basis": "saved per-seed pooled out-of-fold scores and their saved five-seed means",
            "metric_directions": {"r2": "higher is better", "mae": "lower is better", "rmse": "lower is better"},
            "spearman": "correlation of row and group average-tie ranks within the same metric and screen subset",
            "subset": "whole-screen finite numeric mean yield >= 0.05, including zeros; no row filtering",
            "selection": "retrospective top-k screen selection by descending saved row R2 or grouped R2",
            "selection_evaluation": "equal-weight mean saved grouped R2 of the selected screens; group choice minus row choice",
            "matching_seed_selection": "selection and grouped-score evaluation use the same saved seed",
            "tie_break": "exact score ties in top-k selection are broken alphabetically by source_name",
            "discordance": "an unordered screen pair has opposite strict R2 ordering under row and group splits; tied pairs are not discordant",
            "leave_one_screen_out": "omit each retained screen in turn and recompute row/group R2 Spearman across the remaining ten screens",
            "seed_ranges": "minimum and maximum across five overlapping saved seed estimates; descriptive ranges, not confidence intervals",
            "analysis_operations": "rank, compare and average saved scores; no model fits, predictions or loss regeneration",
        },
        "population": {
            "seeds": seeds,
            "all_screen_count": len(sources),
            "all_screen_names": [source["source_name"] for source in sources],
            "mean_yield_threshold_fraction": MEAN_YIELD_THRESHOLD,
            "retained_screen_count": len(retained),
            "retained_screen_names": [source["source_name"] for source in retained],
            "excluded_screen_names": [source["source_name"] for source in sources if source not in retained],
        },
        "metric_rank_concordance": {
            "all_screens": metric_concordance(sources, seeds),
            "mean_yield_ge_0_05": metric_concordance(retained, seeds),
        },
        "retrospective_selection": retrospective_selection(retained, seeds),
        "r2_discordant_pairs": r2_discordance(retained, seeds),
        "leave_one_screen_out_r2": leave_one_screen_out_r2(retained, seeds),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT, help="Saved pooled-score JSON")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Diagnostics JSON")
    args = parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        raise ValueError("diagnostics output cannot overwrite the saved-score input")
    content = args.input.read_bytes()
    input_sha256 = sha256_bytes(content)
    diagnostics = build_diagnostics(json.loads(content), args.input, input_sha256)
    if sha256_bytes(args.input.read_bytes()) != input_sha256:
        raise ValueError("saved-score input changed during diagnostics")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    top_three = diagnostics["retrospective_selection"]["screen_mean"]["by_k"][2]
    print(json.dumps({
        "output": args.output.resolve().as_posix(),
        "screens": diagnostics["population"]["all_screen_count"],
        "retained_screens": diagnostics["population"]["retained_screen_count"],
        "screen_mean_rho": {subset: record["screen_mean"] for subset, record in diagnostics["metric_rank_concordance"].items()},
        "top_three": top_three,
        "discordant_pairs": diagnostics["r2_discordant_pairs"]["screen_mean"]["discordant_count"],
        "leave_one_screen_out_rho_range": diagnostics["leave_one_screen_out_r2"]["screen_mean"]["rho_range"],
    }, indent=2))


if __name__ == "__main__":
    main()
