from __future__ import annotations

import argparse
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from test_final_dag import (
    DEFAULT_DAG_PATH,
    DEFAULT_DATASET_PATH,
    adjustment_set_backdoor,
    build_graph,
    load_dag_config,
    read_dataset,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "education_effect"

TREATMENT = "escolaridade_grupo"
OUTCOME = "morte_evitavel"
TREATMENT_LEVELS = ["baixa", "media", "alta"]
REFERENCE_LEVEL = "baixa"
CONTRASTS = [
    ("baixa", "media"),
    ("baixa", "alta"),
    ("media", "alta"),
]


def prepare_analysis_data(
    dataset_path: Path,
    dag_config: dict,
    graph,
    max_rows: int | None,
    random_seed: int,
) -> tuple[pd.DataFrame, list[str]]:
    adjustment_set = sorted(adjustment_set_backdoor(graph, TREATMENT, OUTCOME) or [])
    required_columns = sorted(set([TREATMENT, OUTCOME] + adjustment_set + ["idade_grupo"]))

    df = read_dataset(dataset_path, set(graph.nodes()) | set(required_columns))
    df = df[required_columns].dropna().copy()

    df[OUTCOME] = pd.to_numeric(df[OUTCOME], errors="coerce")
    df = df[df[OUTCOME].isin([0, 1])].copy()
    df["morte_evitavel_binaria"] = df[OUTCOME].astype(int)

    df = df[df[TREATMENT].isin(TREATMENT_LEVELS)].copy()
    df[TREATMENT] = pd.Categorical(df[TREATMENT], categories=TREATMENT_LEVELS, ordered=True)

    for column in adjustment_set:
        df[column] = df[column].astype("string")

    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_seed).copy()

    return df, adjustment_set


def build_model(df: pd.DataFrame, adjustment_set: list[str]) -> Pipeline:
    features = [TREATMENT] + adjustment_set
    categories = [TREATMENT_LEVELS]
    dropped_categories = [REFERENCE_LEVEL]

    for column in adjustment_set:
        column_categories = sorted(df[column].dropna().astype(str).unique().tolist())
        categories.append(column_categories)
        dropped_categories.append(column_categories[0] if column_categories else None)

    encoder = OneHotEncoder(
        categories=categories,
        drop=dropped_categories,
        handle_unknown="ignore",
        sparse_output=True,
    )

    preprocessor = ColumnTransformer(
        transformers=[("categorical", encoder, features)],
        remainder="drop",
    )

    model = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=1000,
        n_jobs=None,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )


def fit_model(df: pd.DataFrame, adjustment_set: list[str]) -> Pipeline:
    features = [TREATMENT] + adjustment_set
    y = df["morte_evitavel_binaria"].astype(int)
    model = build_model(df, adjustment_set)
    model.fit(df[features], y)
    return model


def standardized_risks(model: Pipeline, df: pd.DataFrame, adjustment_set: list[str]) -> pd.DataFrame:
    rows = []
    features = [TREATMENT] + adjustment_set

    for level in TREATMENT_LEVELS:
        counterfactual = df[features].copy()
        counterfactual[TREATMENT] = level
        predicted_risk = model.predict_proba(counterfactual)[:, 1]

        rows.append(
            {
                "treatment": TREATMENT,
                "level": level,
                "standardized_risk": float(predicted_risk.mean()),
                "n_rows_standardized_over": len(counterfactual),
            }
        )

    return pd.DataFrame(rows)


def pairwise_effects(risks: pd.DataFrame) -> pd.DataFrame:
    risk_by_level = dict(zip(risks["level"], risks["standardized_risk"]))
    rows = []

    for reference in TREATMENT_LEVELS:
        for comparison in TREATMENT_LEVELS:
            if reference == comparison:
                continue

            ref_risk = risk_by_level[reference]
            comp_risk = risk_by_level[comparison]
            ref_odds = ref_risk / (1 - ref_risk)
            comp_odds = comp_risk / (1 - comp_risk)

            rows.append(
                {
                    "comparison": f"{comparison}_vs_{reference}",
                    "reference_level": reference,
                    "comparison_level": comparison,
                    "risk_reference": ref_risk,
                    "risk_comparison": comp_risk,
                    "risk_difference": comp_risk - ref_risk,
                    "risk_ratio": comp_risk / ref_risk if ref_risk > 0 else np.nan,
                    "odds_ratio_from_standardized_risks": comp_odds / ref_odds if ref_odds > 0 else np.nan,
                }
            )

    return pd.DataFrame(rows)


def common_support_mask(
    df: pd.DataFrame,
    adjustment_set: list[str],
    level_a: str,
    level_b: str,
    min_cell_count: int,
) -> pd.Series:
    pair = df[df[TREATMENT].isin([level_a, level_b])].copy()
    counts = (
        pair.groupby(adjustment_set + [TREATMENT], dropna=False, observed=True)
        .size()
        .reset_index(name="n")
    )

    wide = counts.pivot_table(
        index=adjustment_set,
        columns=TREATMENT,
        values="n",
        fill_value=0,
        aggfunc="sum",
        observed=True,
    ).reset_index()

    for level in [level_a, level_b]:
        if level not in wide.columns:
            wide[level] = 0

    supported = wide[(wide[level_a] >= min_cell_count) & (wide[level_b] >= min_cell_count)]
    supported_keys = set(map(tuple, supported[adjustment_set].astype(str).to_numpy()))
    row_keys = list(map(tuple, df[adjustment_set].astype(str).to_numpy()))

    return pd.Series([key in supported_keys for key in row_keys], index=df.index)


def support_keys_for_pair(
    df: pd.DataFrame,
    adjustment_set: list[str],
    reference_level: str,
    comparison_level: str,
    min_cell_count: int,
) -> set[tuple[str, ...]]:
    mask = common_support_mask(
        df,
        adjustment_set=adjustment_set,
        level_a=reference_level,
        level_b=comparison_level,
        min_cell_count=min_cell_count,
    )
    return set(map(tuple, df.loc[mask, adjustment_set].astype(str).to_numpy()))


def build_support_key_map(
    df: pd.DataFrame,
    adjustment_set: list[str],
    min_cell_count: int,
) -> dict[str, set[tuple[str, ...]]]:
    return {
        f"{comparison_level}_vs_{reference_level}": support_keys_for_pair(
            df,
            adjustment_set=adjustment_set,
            reference_level=reference_level,
            comparison_level=comparison_level,
            min_cell_count=min_cell_count,
        )
        for reference_level, comparison_level in CONTRASTS
    }


def pairwise_common_support_diagnostics(
    df: pd.DataFrame,
    adjustment_set: list[str],
    min_cell_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    cell_rows = []

    for level_a, level_b in combinations(TREATMENT_LEVELS, 2):
        pair = df[df[TREATMENT].isin([level_a, level_b])].copy()
        counts = (
            pair.groupby(adjustment_set + [TREATMENT], dropna=False, observed=True)
            .size()
            .reset_index(name="n")
        )

        wide = counts.pivot_table(
            index=adjustment_set,
            columns=TREATMENT,
            values="n",
            fill_value=0,
            aggfunc="sum",
            observed=True,
        ).reset_index()

        for level in [level_a, level_b]:
            if level not in wide.columns:
                wide[level] = 0

        wide["contrast"] = f"{level_b}_vs_{level_a}"
        wide["level_a"] = level_a
        wide["level_b"] = level_b
        wide["pair_stratum_n"] = wide[level_a] + wide[level_b]
        wide["has_pair_common_support"] = (wide[level_a] >= min_cell_count) & (wide[level_b] >= min_cell_count)
        wide["min_pair_cell_n"] = wide[[level_a, level_b]].min(axis=1)
        cell_rows.append(wide)

        n_pair = len(pair)
        supported_keys = set(
            map(tuple, wide.loc[wide["has_pair_common_support"], adjustment_set].astype(str).to_numpy())
        )
        pair_keys = list(map(tuple, pair[adjustment_set].astype(str).to_numpy()))
        n_pair_in_common_support = sum(key in supported_keys for key in pair_keys)

        summary_rows.append(
            {
                "contrast": f"{level_b}_vs_{level_a}",
                "level_a": level_a,
                "level_b": level_b,
                "adjustment_set": "|".join(adjustment_set),
                "min_cell_count": min_cell_count,
                "n_pair_rows": n_pair,
                "n_pair_rows_in_common_support": int(n_pair_in_common_support),
                "pct_pair_rows_in_common_support": n_pair_in_common_support / n_pair if n_pair else 0.0,
                "n_strata_pair_observed": len(wide),
                "n_strata_pair_common_support": int(wide["has_pair_common_support"].sum()),
                "pct_strata_pair_common_support": float(wide["has_pair_common_support"].mean()) if len(wide) else 0.0,
                "min_pair_stratum_n": int(wide["pair_stratum_n"].min()) if len(wide) else 0,
                "median_pair_stratum_n": float(wide["pair_stratum_n"].median()) if len(wide) else 0.0,
            }
        )

    summary = pd.DataFrame(summary_rows)
    cells = pd.concat(cell_rows, ignore_index=True) if cell_rows else pd.DataFrame()
    return summary, cells


def pairwise_common_support_effects(
    model: Pipeline,
    df: pd.DataFrame,
    adjustment_set: list[str],
    min_cell_count: int,
    support_key_map: dict[str, set[tuple[str, ...]]] | None = None,
) -> pd.DataFrame:
    rows = []
    features = [TREATMENT] + adjustment_set

    for reference, comparison in CONTRASTS:
        contrast = f"{comparison}_vs_{reference}"
        pair_df = df[df[TREATMENT].isin([reference, comparison])].copy()
        n_pair_before_support = len(pair_df)

        support_keys = None if support_key_map is None else support_key_map.get(contrast)
        if support_keys is None:
            support_keys = support_keys_for_pair(
                pair_df,
                adjustment_set=adjustment_set,
                reference_level=reference,
                comparison_level=comparison,
                min_cell_count=min_cell_count,
            )

        pair_keys = list(map(tuple, pair_df[adjustment_set].astype(str).to_numpy()))
        support_df = pair_df.loc[[key in support_keys for key in pair_keys], features].copy()
        n_removed_support = n_pair_before_support - len(support_df)

        if support_df.empty:
            rows.append(
                {
                    "contrast": contrast,
                    "reference_level": reference,
                    "comparison_level": comparison,
                    "n_used": 0,
                    "n_removed_no_common_support": n_removed_support,
                    "n_rows_standardized_over": 0,
                    "risk_reference": np.nan,
                    "risk_comparison": np.nan,
                    "risk_difference": np.nan,
                    "risk_ratio": np.nan,
                    "odds_ratio_from_standardized_risks": np.nan,
                }
            )
            continue

        reference_df = support_df.copy()
        comparison_df = support_df.copy()
        reference_df[TREATMENT] = reference
        comparison_df[TREATMENT] = comparison

        reference_risk = float(model.predict_proba(reference_df)[:, 1].mean())
        comparison_risk = float(model.predict_proba(comparison_df)[:, 1].mean())
        reference_odds = reference_risk / (1 - reference_risk)
        comparison_odds = comparison_risk / (1 - comparison_risk)

        rows.append(
            {
                "contrast": contrast,
                "reference_level": reference,
                "comparison_level": comparison,
                "n_used": len(support_df),
                "n_removed_no_common_support": n_removed_support,
                "n_rows_standardized_over": len(support_df),
                "risk_reference": reference_risk,
                "risk_comparison": comparison_risk,
                "risk_difference": comparison_risk - reference_risk,
                "risk_ratio": comparison_risk / reference_risk if reference_risk > 0 else np.nan,
                "odds_ratio_from_standardized_risks": comparison_odds / reference_odds if reference_odds > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def bootstrap_common_support_effects(
    df: pd.DataFrame,
    adjustment_set: list[str],
    min_cell_count: int,
    n_iterations: int,
    sample_size: int | None,
    evaluation_sample_size: int | None,
    random_seed: int,
    support_key_map: dict[str, set[tuple[str, ...]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_iterations <= 0:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    rows = []
    n = len(df)
    effective_sample_size = n if sample_size is None or sample_size <= 0 else min(sample_size, n)
    effective_evaluation_size = n if evaluation_sample_size is None or evaluation_sample_size <= 0 else min(evaluation_sample_size, n)
    evaluation_df = df.sample(n=effective_evaluation_size, replace=False, random_state=random_seed).copy()

    for iteration in range(1, n_iterations + 1):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        boot = df.sample(n=effective_sample_size, replace=True, random_state=seed).copy()
        boot_model = fit_model(boot, adjustment_set)
        boot_effects = pairwise_common_support_effects(
            boot_model,
            evaluation_df,
            adjustment_set=adjustment_set,
            min_cell_count=min_cell_count,
            support_key_map=support_key_map,
        )
        boot_effects.insert(0, "bootstrap_iteration", iteration)
        boot_effects.insert(1, "bootstrap_sample_size", effective_sample_size)
        boot_effects.insert(2, "bootstrap_evaluation_sample_size", effective_evaluation_size)
        rows.append(boot_effects)

        if iteration == 1 or iteration % 10 == 0 or iteration == n_iterations:
            print(f"Bootstrap {iteration}/{n_iterations} concluido.")

    draws = pd.concat(rows, ignore_index=True)
    metric_cols = ["risk_reference", "risk_comparison", "risk_difference", "risk_ratio", "odds_ratio_from_standardized_risks"]
    summary_rows = []

    for contrast, group in draws.groupby("contrast", dropna=False):
        row = {
            "contrast": contrast,
            "n_bootstrap_iterations": n_iterations,
            "bootstrap_sample_size": effective_sample_size,
            "bootstrap_evaluation_sample_size": effective_evaluation_size,
            "n_successful_draws": int(group["risk_difference"].notna().sum()),
        }

        first = group.iloc[0]
        row["reference_level"] = first["reference_level"]
        row["comparison_level"] = first["comparison_level"]

        for metric in metric_cols:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_median"] = float(values.median()) if len(values) else np.nan
            row[f"{metric}_ci_low"] = float(values.quantile(0.025)) if len(values) else np.nan
            row[f"{metric}_ci_high"] = float(values.quantile(0.975)) if len(values) else np.nan

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    return draws, summary


def model_coefficients(model: Pipeline) -> pd.DataFrame:
    preprocessor = model.named_steps["preprocess"]
    logistic = model.named_steps["model"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = logistic.coef_[0]

    rows = [
        {
            "feature": "intercept",
            "log_odds_coefficient": float(logistic.intercept_[0]),
            "odds_ratio": float(np.exp(logistic.intercept_[0])),
        }
    ]

    rows.extend(
        {
            "feature": feature,
            "log_odds_coefficient": float(coef),
            "odds_ratio": float(np.exp(coef)),
        }
        for feature, coef in zip(feature_names, coefficients)
    )

    return pd.DataFrame(rows)


def sample_summary(df: pd.DataFrame, adjustment_set: list[str], dag_path: Path, dataset_path: Path) -> pd.DataFrame:
    rows = [
        {
            "dataset_path": str(dataset_path),
            "dag_path": str(dag_path),
            "estimand": "efeito_total_escolaridade_sobre_morte_evitavel",
            "outcome_original": OUTCOME,
            "outcome_modeled": "morte_evitavel_binaria",
            "outcome_filter": "mantem morte_evitavel em {0, 1}; exclui 2 (mal definida)",
            "treatment": TREATMENT,
            "treatment_levels": "|".join(TREATMENT_LEVELS),
            "reference_level": REFERENCE_LEVEL,
            "adjustment_set": "|".join(adjustment_set),
            "n_rows": len(df),
            "outcome_mean": float(df["morte_evitavel_binaria"].mean()) if len(df) else np.nan,
        }
    ]
    return pd.DataFrame(rows)


def treatment_distribution(df: pd.DataFrame) -> pd.DataFrame:
    table = df.groupby(TREATMENT, dropna=False, observed=False).size().reset_index(name="n")
    table["share"] = table["n"] / table["n"].sum()
    return table


def positivity_for_estimation(df: pd.DataFrame, adjustment_set: list[str], min_cell_count: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = (
        df.groupby(adjustment_set + [TREATMENT], dropna=False, observed=True)
        .size()
        .reset_index(name="n")
    )

    strata = (
        counts[counts["n"] > 0]
        .groupby(adjustment_set, dropna=False, observed=True)
        .agg(
            stratum_n=("n", "sum"),
            n_present_treatment_levels=(TREATMENT, "nunique"),
            min_observed_treatment_cell_n=("n", "min"),
        )
        .reset_index()
    )
    strata["has_all_treatment_levels"] = strata["n_present_treatment_levels"] == len(TREATMENT_LEVELS)
    strata["has_small_observed_cell"] = strata["min_observed_treatment_cell_n"] < min_cell_count

    summary = pd.DataFrame(
        [
            {
                "adjustment_set": "|".join(adjustment_set),
                "n_strata": len(strata),
                "min_cell_count": min_cell_count,
                "n_strata_missing_treatment_level": int((~strata["has_all_treatment_levels"]).sum()),
                "pct_strata_missing_treatment_level": float((~strata["has_all_treatment_levels"]).mean()),
                "n_strata_with_small_observed_cell": int(strata["has_small_observed_cell"].sum()),
                "pct_strata_with_small_observed_cell": float(strata["has_small_observed_cell"].mean()),
                "min_stratum_n": int(strata["stratum_n"].min()),
                "median_stratum_n": float(strata["stratum_n"].median()),
            }
        ]
    )

    return summary, counts


def write_run_config(args: argparse.Namespace, adjustment_set: list[str], output_dir: Path) -> None:
    config = {
        "dag": str(args.dag),
        "dataset": str(args.dataset),
        "output_dir": str(output_dir),
        "max_rows": args.max_rows,
        "random_seed": args.random_seed,
        "min_cell_count": args.min_cell_count,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_sample_size": args.bootstrap_sample_size,
        "bootstrap_evaluation_sample_size": args.bootstrap_evaluation_sample_size,
        "treatment": TREATMENT,
        "outcome": OUTCOME,
        "adjustment_set": adjustment_set,
        "support_definition": "pairwise_common_support_keys_from_full_analysis_base",
        "evaluation_population": "observed_pair_levels_with_fixed_common_support",
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima o efeito total de escolaridade_grupo sobre morte_evitavel via g-computation."
    )
    parser.add_argument("--dag", type=Path, default=DEFAULT_DAG_PATH, help="Arquivo JSON com o DAG final.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="CSV tratado do SIM.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretorio de saida.")
    parser.add_argument("--max-rows", type=int, default=0, help="Subamostra opcional para teste; 0 usa a base completa.")
    parser.add_argument("--random-seed", type=int, default=42, help="Semente para subamostragem.")
    parser.add_argument("--min-cell-count", type=int, default=20, help="Celula minima para diagnostico de positividade.")
    parser.add_argument("--bootstrap-iterations", type=int, default=0, help="Numero de reamostragens bootstrap; 0 desativa.")
    parser.add_argument("--bootstrap-sample-size", type=int, default=100000, help="Tamanho de cada amostra bootstrap; 0 usa a base de analise inteira.")
    parser.add_argument("--bootstrap-evaluation-sample-size", type=int, default=100000, help="Tamanho da amostra fixa usada para padronizar cada bootstrap; 0 usa a base de analise inteira.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dag_config = load_dag_config(args.dag)
    graph = build_graph(dag_config["edges"])
    max_rows = None if args.max_rows == 0 else args.max_rows

    df, adjustment_set = prepare_analysis_data(
        dataset_path=args.dataset,
        dag_config=dag_config,
        graph=graph,
        max_rows=max_rows,
        random_seed=args.random_seed,
    )

    if df.empty:
        raise ValueError("A amostra de analise ficou vazia depois dos filtros.")

    model = fit_model(df, adjustment_set)
    risks = standardized_risks(model, df, adjustment_set)
    effects = pairwise_effects(risks)
    coefficients = model_coefficients(model)
    support_key_map = build_support_key_map(
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
    )
    positivity_summary, positivity_cells = positivity_for_estimation(
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
    )
    pairwise_positivity_summary, pairwise_positivity_cells = pairwise_common_support_diagnostics(
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
    )
    common_support_effects = pairwise_common_support_effects(
        model,
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
        support_key_map=support_key_map,
    )
    bootstrap_sample_size = None if args.bootstrap_sample_size == 0 else args.bootstrap_sample_size
    bootstrap_evaluation_sample_size = None if args.bootstrap_evaluation_sample_size == 0 else args.bootstrap_evaluation_sample_size
    bootstrap_draws, bootstrap_summary = bootstrap_common_support_effects(
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
        n_iterations=args.bootstrap_iterations,
        sample_size=bootstrap_sample_size,
        evaluation_sample_size=bootstrap_evaluation_sample_size,
        random_seed=args.random_seed,
        support_key_map=support_key_map,
    )

    sample_summary(df, adjustment_set, args.dag, args.dataset).to_csv(
        args.output_dir / "sample_summary.csv",
        index=False,
    )
    treatment_distribution(df).to_csv(args.output_dir / "treatment_distribution.csv", index=False)
    positivity_summary.to_csv(args.output_dir / "positivity_summary.csv", index=False)
    positivity_cells.to_csv(args.output_dir / "positivity_cells.csv", index=False)
    pairwise_positivity_summary.to_csv(
        args.output_dir / "pairwise_positivity_summary.csv",
        index=False,
    )
    pairwise_positivity_cells.to_csv(
        args.output_dir / "pairwise_positivity_cells.csv",
        index=False,
    )
    risks.to_csv(args.output_dir / "standardized_risks.csv", index=False)
    effects.to_csv(args.output_dir / "effect_estimates.csv", index=False)
    common_support_effects.to_csv(
        args.output_dir / "effect_estimates_common_support.csv",
        index=False,
    )
    if not bootstrap_draws.empty:
        bootstrap_draws.to_csv(
            args.output_dir / "bootstrap_effect_estimates_common_support.csv",
            index=False,
        )
        bootstrap_summary.to_csv(
            args.output_dir / "bootstrap_effect_summary_common_support.csv",
            index=False,
        )
    coefficients.to_csv(args.output_dir / "model_coefficients.csv", index=False)
    write_run_config(args, adjustment_set, args.output_dir)

    print("Estimacao concluida.")
    print(f"Linhas na analise: {len(df):,}")
    print(f"Ajuste: {', '.join(adjustment_set)}")
    print(f"Saidas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
