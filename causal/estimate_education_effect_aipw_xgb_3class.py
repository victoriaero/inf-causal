from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    XGBClassifier = None

from estimate_education_effect import (
    CONTRASTS,
    OUTCOME,
    PROJECT_ROOT,
    TREATMENT,
    TREATMENT_LEVELS,
    prepare_analysis_data,
)
from test_final_dag import DEFAULT_DAG_PATH, DEFAULT_DATASET_PATH, build_graph, load_dag_config


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "education_effect_aipw_xgb_3class"
LEVEL_TO_CODE = {level: index for index, level in enumerate(TREATMENT_LEVELS)}
CODE_TO_LEVEL = {index: level for level, index in LEVEL_TO_CODE.items()}


def category_levels(df: pd.DataFrame, features: list[str]) -> list[list[str]]:
    levels = []
    for feature in features:
        if feature == TREATMENT:
            levels.append(TREATMENT_LEVELS)
        else:
            levels.append(sorted(df[feature].dropna().astype(str).unique().tolist()))
    return levels


def build_xgb_pipeline(
    category_df: pd.DataFrame,
    features: list[str],
    objective: str,
    eval_metric: str,
    random_seed: int,
    num_class: int | None = None,
) -> Pipeline:
    if XGBClassifier is None:
        raise ImportError("xgboost nao esta instalado. Instale com: python3 -m pip install xgboost")

    encoder = OneHotEncoder(
        categories=category_levels(category_df, features),
        drop=None,
        handle_unknown="ignore",
        sparse_output=True,
    )
    params = {
        "objective": objective,
        "eval_metric": eval_metric,
        "n_estimators": 200,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 20,
        "reg_lambda": 1.0,
        "random_state": random_seed,
        "n_jobs": 1,
        "tree_method": "hist",
    }
    if num_class is not None:
        params["num_class"] = num_class

    preprocessor = ColumnTransformer(
        transformers=[("categorical", encoder, features)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", XGBClassifier(**params)),
        ]
    )


def global_support_keys(
    df: pd.DataFrame,
    adjustment_set: list[str],
    min_cell_count: int,
) -> set[tuple[str, ...]]:
    counts = (
        df.groupby(adjustment_set + [TREATMENT], dropna=False, observed=True)
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

    for level in TREATMENT_LEVELS:
        if level not in wide.columns:
            wide[level] = 0

    supported = wide[
        np.logical_and.reduce([wide[level] >= min_cell_count for level in TREATMENT_LEVELS])
    ]
    return set(map(tuple, supported[adjustment_set].astype(str).to_numpy()))


def apply_global_support(
    df: pd.DataFrame,
    adjustment_set: list[str],
    support_keys: set[tuple[str, ...]],
) -> tuple[pd.DataFrame, int]:
    before = len(df)
    row_keys = list(map(tuple, df[adjustment_set].astype(str).to_numpy()))
    supported = df.loc[[key in support_keys for key in row_keys]].copy()
    supported = supported[supported[TREATMENT].isin(TREATMENT_LEVELS)].copy()
    supported["treatment_code"] = supported[TREATMENT].map(LEVEL_TO_CODE).astype(int)
    return supported, before - len(supported)


def global_support_summary(
    df: pd.DataFrame,
    adjustment_set: list[str],
    min_cell_count: int,
    support_keys: set[tuple[str, ...]],
) -> pd.DataFrame:
    supported, n_removed = apply_global_support(df, adjustment_set, support_keys)
    counts = supported.groupby(TREATMENT, dropna=False, observed=False).size().reset_index(name="n")
    counts["share"] = counts["n"] / counts["n"].sum()
    counts.insert(0, "support_type", f"global_three_level_common_support_min_cell_{min_cell_count}")

    summary = pd.DataFrame(
        [
            {
                "support_type": f"global_three_level_common_support_min_cell_{min_cell_count}",
                "adjustment_set": "|".join(adjustment_set),
                "n_rows_before_support": len(df),
                "n_rows_after_support": len(supported),
                "n_removed_no_common_support": n_removed,
                "pct_rows_after_support": len(supported) / len(df) if len(df) else 0.0,
                "n_supported_strata": len(support_keys),
                "min_cell_count": min_cell_count,
            }
        ]
    )
    return summary, counts


def fit_nuisance_models(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    adjustment_set: list[str],
    random_seed: int,
) -> tuple[Pipeline, Pipeline]:
    outcome_features = [TREATMENT] + adjustment_set
    treatment_features = adjustment_set
    category_df = pd.concat([train, eval_df], ignore_index=True, sort=False)

    outcome_model = build_xgb_pipeline(
        category_df=category_df,
        features=outcome_features,
        objective="binary:logistic",
        eval_metric="logloss",
        random_seed=random_seed,
    )
    treatment_model = build_xgb_pipeline(
        category_df=category_df,
        features=treatment_features,
        objective="multi:softprob",
        eval_metric="mlogloss",
        random_seed=random_seed,
        num_class=len(TREATMENT_LEVELS),
    )

    outcome_model.fit(train[outcome_features], train["morte_evitavel_binaria"].astype(int))
    treatment_model.fit(train[treatment_features], train["treatment_code"].astype(int))
    return outcome_model, treatment_model


def predict_outcome_under(
    outcome_model: Pipeline,
    eval_df: pd.DataFrame,
    adjustment_set: list[str],
    treatment_level: str,
) -> np.ndarray:
    features = [TREATMENT] + adjustment_set
    counterfactual = eval_df[features].copy()
    counterfactual[TREATMENT] = treatment_level
    return outcome_model.predict_proba(counterfactual[features])[:, 1]


def estimate_aipw_3class(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    adjustment_set: list[str],
    random_seed: int,
    min_cell_count: int,
    propensity_clip: float,
    support_keys: set[tuple[str, ...]],
    crossfit_folds: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_supported, _ = apply_global_support(train_df, adjustment_set, support_keys)
    eval_supported, n_removed_eval = apply_global_support(eval_df, adjustment_set, support_keys)

    if train_supported.empty or eval_supported.empty:
        raise ValueError("A amostra ficou vazia depois do suporte comum global.")
    if train_supported["treatment_code"].nunique() != len(TREATMENT_LEVELS):
        raise ValueError("A amostra de treino nao contem os tres niveis no suporte comum global.")

    observed_treatment = eval_supported["treatment_code"].to_numpy()
    outcome = eval_supported["morte_evitavel_binaria"].to_numpy()
    fold_count = 1
    propensities = np.empty((len(eval_supported), len(TREATMENT_LEVELS)), dtype=float)
    mu_by_level = {level: np.empty(len(eval_supported), dtype=float) for level in TREATMENT_LEVELS}

    if crossfit_folds > 1 and train_df is eval_df:
        fold_count = min(crossfit_folds, int(eval_supported["treatment_code"].value_counts().min()))
        if fold_count < 2:
            fold_count = 1

    if fold_count > 1:
        splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
        y_strata = eval_supported["treatment_code"].astype(int).to_numpy()

        for fold_number, (train_index, eval_index) in enumerate(splitter.split(eval_supported, y_strata), start=1):
            print(f"Cross-fit xgb_3class: fold {fold_number}/{fold_count}.")
            train_fold = eval_supported.iloc[train_index].copy()
            eval_fold = eval_supported.iloc[eval_index].copy()
            if train_fold["treatment_code"].nunique() != len(TREATMENT_LEVELS):
                raise ValueError("Fold invalido: faltou algum nivel de escolaridade no treino.")
            if train_fold["morte_evitavel_binaria"].nunique() < 2:
                raise ValueError("Fold invalido: faltou variacao no desfecho.")

            outcome_model, treatment_model = fit_nuisance_models(
                train=train_fold,
                eval_df=eval_supported,
                adjustment_set=adjustment_set,
                random_seed=random_seed,
            )
            fold_propensities = treatment_model.predict_proba(eval_fold[adjustment_set])
            if fold_propensities.shape[1] != len(TREATMENT_LEVELS):
                raise ValueError("O modelo de propensao nao retornou probabilidades para os tres niveis.")

            propensities[eval_index] = fold_propensities
            for level in TREATMENT_LEVELS:
                mu_by_level[level][eval_index] = predict_outcome_under(
                    outcome_model,
                    eval_fold,
                    adjustment_set=adjustment_set,
                    treatment_level=level,
                )
    else:
        outcome_model, treatment_model = fit_nuisance_models(
            train=train_supported,
            eval_df=eval_supported,
            adjustment_set=adjustment_set,
            random_seed=random_seed,
        )
        propensities = treatment_model.predict_proba(eval_supported[adjustment_set])
        if propensities.shape[1] != len(TREATMENT_LEVELS):
            raise ValueError("O modelo de propensao nao retornou probabilidades para os tres niveis.")

        for level in TREATMENT_LEVELS:
            mu_by_level[level] = predict_outcome_under(
                outcome_model,
                eval_supported,
                adjustment_set=adjustment_set,
                treatment_level=level,
            )

    clipped_propensities = np.clip(propensities, propensity_clip, 1 - propensity_clip)
    risk_rows = []
    diagnostic_rows = []

    for level_index, level in enumerate(TREATMENT_LEVELS):
        mu = mu_by_level[level]
        treatment_indicator = (observed_treatment == level_index).astype(int)
        p_level = clipped_propensities[:, level_index]
        psi = mu + treatment_indicator * (outcome - mu) / p_level
        risk_rows.append(
            {
                "level": level,
                "treatment_code": level_index,
                "risk": float(np.mean(psi)),
                "n_used": len(eval_supported),
                "n_removed_no_common_support": n_removed_eval,
                "support_type": f"global_three_level_common_support_min_cell_{min_cell_count}",
                "model_type": "xgb_3class",
                "crossfit_folds": fold_count,
            }
        )
        raw_p_level = propensities[:, level_index]
        diagnostic_rows.append(
            {
                "level": level,
                "treatment_code": level_index,
                "propensity_mean": float(np.mean(raw_p_level)),
                "propensity_min": float(np.min(raw_p_level)),
                "propensity_p01": float(np.quantile(raw_p_level, 0.01)),
                "propensity_p05": float(np.quantile(raw_p_level, 0.05)),
                "propensity_p50": float(np.quantile(raw_p_level, 0.50)),
                "propensity_p95": float(np.quantile(raw_p_level, 0.95)),
                "propensity_p99": float(np.quantile(raw_p_level, 0.99)),
                "propensity_max": float(np.max(raw_p_level)),
                "propensity_extreme_share": float(np.mean((raw_p_level < 0.01) | (raw_p_level > 0.99))),
                "n_used": len(eval_supported),
                "crossfit_folds": fold_count,
            }
        )

    risks = pd.DataFrame(risk_rows)
    risk_by_level = dict(zip(risks["level"], risks["risk"]))
    contrast_rows = []
    for reference_level, comparison_level in CONTRASTS:
        risk_reference = risk_by_level[reference_level]
        risk_comparison = risk_by_level[comparison_level]
        contrast_rows.append(
            {
                "contrast": f"{comparison_level}_vs_{reference_level}",
                "treated_level": comparison_level,
                "reference_level": reference_level,
                "n_used": len(eval_supported),
                "n_removed_no_common_support": n_removed_eval,
                "risk_treated": risk_comparison,
                "risk_reference": risk_reference,
                "risk_difference": risk_comparison - risk_reference,
                "risk_ratio": risk_comparison / risk_reference if risk_reference > 0 else np.nan,
                "model_type": "xgb_3class",
                "support_type": f"global_three_level_common_support_min_cell_{min_cell_count}",
                "crossfit_folds": fold_count,
            }
        )

    return risks, pd.DataFrame(contrast_rows), pd.DataFrame(diagnostic_rows)


def bootstrap_aipw_3class(
    df: pd.DataFrame,
    adjustment_set: list[str],
    random_seed: int,
    min_cell_count: int,
    propensity_clip: float,
    n_iterations: int,
    bootstrap_sample_size: int | None,
    evaluation_sample_size: int | None,
    support_keys: set[tuple[str, ...]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if n_iterations <= 0:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(random_seed)
    n = len(df)
    train_n = n if bootstrap_sample_size is None or bootstrap_sample_size <= 0 else min(bootstrap_sample_size, n)
    eval_n = n if evaluation_sample_size is None or evaluation_sample_size <= 0 else min(evaluation_sample_size, n)
    eval_df = df.sample(n=eval_n, replace=False, random_state=random_seed).copy()
    draw_rows = []

    for iteration in range(1, n_iterations + 1):
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        train_df = df.sample(n=train_n, replace=True, random_state=seed).copy()
        _, effects, _ = estimate_aipw_3class(
            train_df=train_df,
            eval_df=eval_df,
            adjustment_set=adjustment_set,
            random_seed=seed,
            min_cell_count=min_cell_count,
            propensity_clip=propensity_clip,
            support_keys=support_keys,
        )
        effects.insert(0, "bootstrap_iteration", iteration)
        effects.insert(1, "bootstrap_sample_size", train_n)
        effects.insert(2, "bootstrap_evaluation_sample_size", eval_n)
        draw_rows.append(effects)

        if iteration == 1 or iteration % 10 == 0 or iteration == n_iterations:
            print(f"Bootstrap AIPW XGB 3 classes {iteration}/{n_iterations} concluido.")

    draws = pd.concat(draw_rows, ignore_index=True)
    summary_rows = []
    for contrast, group in draws.groupby("contrast", dropna=False):
        valid = group.dropna(subset=["risk_difference", "risk_ratio"])
        first = group.iloc[0]
        summary_rows.append(
            {
                "contrast": contrast,
                "treated_level": first["treated_level"],
                "reference_level": first["reference_level"],
                "estimate_risk_difference": float(valid["risk_difference"].median()) if len(valid) else np.nan,
                "ci95_low_risk_difference": float(valid["risk_difference"].quantile(0.025)) if len(valid) else np.nan,
                "ci95_high_risk_difference": float(valid["risk_difference"].quantile(0.975)) if len(valid) else np.nan,
                "estimate_risk_ratio": float(valid["risk_ratio"].median()) if len(valid) else np.nan,
                "ci95_low_risk_ratio": float(valid["risk_ratio"].quantile(0.025)) if len(valid) else np.nan,
                "ci95_high_risk_ratio": float(valid["risk_ratio"].quantile(0.975)) if len(valid) else np.nan,
                "n_valid_bootstrap_draws": len(valid),
                "n_bootstrap_iterations": n_iterations,
                "model_type": "xgb_3class",
            }
        )

    return draws, pd.DataFrame(summary_rows)


def sample_summary(df: pd.DataFrame, adjustment_set: list[str], args: argparse.Namespace) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_path": str(args.dataset),
                "dag_path": str(args.dag),
                "estimator": "AIPW",
                "model_type": "xgb_3class",
                "outcome_filter": "mantem morte_evitavel em {0, 1}; exclui 2 (mal definida)",
                "treatment": TREATMENT,
                "treatment_levels": "|".join(TREATMENT_LEVELS),
                "adjustment_set": "|".join(adjustment_set),
                "n_rows": len(df),
                "outcome_mean": float(df["morte_evitavel_binaria"].mean()) if len(df) else np.nan,
                "min_cell_count": args.min_cell_count,
                "propensity_clip": args.propensity_clip,
                "support_definition": "global_three_level_common_support_keys_from_full_analysis_base",
                "evaluation_population": "all_observed_treatment_levels_with_global_common_support",
                "treatment_encoding": "multiclass_three_level_treatment",
                "crossfit_folds": args.crossfit_folds,
            }
        ]
    )


def write_run_config(args: argparse.Namespace, adjustment_set: list[str]) -> None:
    config = {
        "dataset": str(args.dataset),
        "dag": str(args.dag),
        "output_dir": str(args.output_dir),
        "model_type": "xgb_3class",
        "max_rows": args.max_rows,
        "random_seed": args.random_seed,
        "min_cell_count": args.min_cell_count,
        "propensity_clip": args.propensity_clip,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_sample_size": args.bootstrap_sample_size,
        "bootstrap_evaluation_sample_size": args.bootstrap_evaluation_sample_size,
        "crossfit_folds": args.crossfit_folds,
        "adjustment_set": adjustment_set,
        "support_definition": "global_three_level_common_support_keys_from_full_analysis_base",
        "evaluation_population": "all_observed_treatment_levels_with_global_common_support",
        "treatment_encoding": "multiclass_three_level_treatment",
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima escolaridade em tres classes via AIPW com XGBoost e suporte comum global."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dag", type=Path, default=DEFAULT_DAG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-cell-count", type=int, default=20)
    parser.add_argument("--propensity-clip", type=float, default=0.01)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--bootstrap-sample-size", type=int, default=100000)
    parser.add_argument("--bootstrap-evaluation-sample-size", type=int, default=100000)
    parser.add_argument("--crossfit-folds", type=int, default=1)
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

    support_keys = global_support_keys(
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
    )
    support_summary, support_treatment_distribution = global_support_summary(
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
        support_keys=support_keys,
    )

    risks, effects, diagnostics = estimate_aipw_3class(
        train_df=df,
        eval_df=df,
        adjustment_set=adjustment_set,
        random_seed=args.random_seed,
        min_cell_count=args.min_cell_count,
        propensity_clip=args.propensity_clip,
        support_keys=support_keys,
        crossfit_folds=args.crossfit_folds,
    )

    bootstrap_sample_size = None if args.bootstrap_sample_size == 0 else args.bootstrap_sample_size
    evaluation_sample_size = None if args.bootstrap_evaluation_sample_size == 0 else args.bootstrap_evaluation_sample_size
    bootstrap_draws, bootstrap_summary = bootstrap_aipw_3class(
        df=df,
        adjustment_set=adjustment_set,
        random_seed=args.random_seed,
        min_cell_count=args.min_cell_count,
        propensity_clip=args.propensity_clip,
        n_iterations=args.bootstrap_iterations,
        bootstrap_sample_size=bootstrap_sample_size,
        evaluation_sample_size=evaluation_sample_size,
        support_keys=support_keys,
    )

    sample_summary(df, adjustment_set, args).to_csv(args.output_dir / "aipw_3class_sample_summary.csv", index=False)
    support_summary.to_csv(args.output_dir / "aipw_3class_global_support_summary.csv", index=False)
    support_treatment_distribution.to_csv(
        args.output_dir / "aipw_3class_global_support_treatment_distribution.csv",
        index=False,
    )
    risks.to_csv(args.output_dir / "aipw_3class_risks_global_support.csv", index=False)
    effects.to_csv(args.output_dir / "aipw_3class_effect_estimates_global_support.csv", index=False)
    diagnostics.to_csv(args.output_dir / "aipw_3class_propensity_diagnostics.csv", index=False)

    if not bootstrap_draws.empty:
        bootstrap_draws.to_csv(
            args.output_dir / "aipw_3class_bootstrap_effect_estimates_global_support.csv",
            index=False,
        )
        bootstrap_summary.to_csv(
            args.output_dir / "aipw_3class_bootstrap_summary_global_support.csv",
            index=False,
        )

    write_run_config(args, adjustment_set)

    print("Estimacao AIPW XGB 3 classes concluida.")
    print(f"Linhas na analise: {len(df):,}")
    print(f"Estratos com suporte global: {len(support_keys):,}")
    print(f"Ajuste: {', '.join(adjustment_set)}")
    print(f"Saidas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
