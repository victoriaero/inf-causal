from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover - handled at runtime with a clear message.
    XGBClassifier = None

from estimate_education_effect import (
    OUTCOME,
    PROJECT_ROOT,
    TREATMENT,
    TREATMENT_LEVELS,
    common_support_mask,
    prepare_analysis_data,
)
from test_final_dag import DEFAULT_DAG_PATH, DEFAULT_DATASET_PATH, build_graph, load_dag_config


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "education_effect_aipw"
GCOMP_RESULTS_PATH = PROJECT_ROOT / "causal" / "output" / "education_effect" / "effect_estimates_common_support.csv"

CONTRASTS = [
    ("baixa", "media"),
    ("baixa", "alta"),
    ("media", "alta"),
]


def model_features(include_treatment: bool, adjustment_set: list[str]) -> list[str]:
    return (["binary_treatment"] if include_treatment else []) + adjustment_set


def category_levels(df: pd.DataFrame, features: list[str]) -> list[list[str]]:
    levels = []
    for feature in features:
        if feature == "binary_treatment":
            levels.append([0, 1])
        else:
            levels.append(sorted(df[feature].dropna().astype(str).unique().tolist()))
    return levels


def build_classifier(
    category_df: pd.DataFrame,
    features: list[str],
    model_type: str,
    random_seed: int,
) -> Pipeline:
    if model_type == "logistic":
        encoder = OneHotEncoder(
            categories=category_levels(category_df, features),
            drop="first",
            handle_unknown="ignore",
            sparse_output=True,
        )
        classifier = LogisticRegression(max_iter=1000, solver="lbfgs")
    elif model_type == "hgb":
        encoder = OrdinalEncoder(
            categories=category_levels(category_df, features),
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        )
        classifier = HistGradientBoostingClassifier(
            categorical_features=list(range(len(features))),
            random_state=random_seed,
            learning_rate=0.05,
            max_iter=200,
        )
    elif model_type == "xgb":
        if XGBClassifier is None:
            raise ImportError("xgboost nao esta instalado. Instale com: python3 -m pip install xgboost")
        encoder = OneHotEncoder(
            categories=category_levels(category_df, features),
            drop=None,
            handle_unknown="ignore",
            sparse_output=True,
        )
        classifier = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=20,
            reg_lambda=1.0,
            random_state=random_seed,
            n_jobs=1,
            tree_method="hist",
        )
    else:
        raise ValueError(f"model_type invalido: {model_type}")

    preprocessor = ColumnTransformer(
        transformers=[("categorical", encoder, features)],
        remainder="drop",
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", classifier),
        ]
    )


def fit_nuisance_models(
    train: pd.DataFrame,
    category_df: pd.DataFrame,
    adjustment_set: list[str],
    model_type: str,
    random_seed: int,
) -> tuple[Pipeline, Pipeline]:
    outcome_features = model_features(include_treatment=True, adjustment_set=adjustment_set)
    treatment_features = model_features(include_treatment=False, adjustment_set=adjustment_set)

    outcome_model = build_classifier(category_df, outcome_features, model_type, random_seed)
    treatment_model = build_classifier(category_df, treatment_features, model_type, random_seed)

    outcome_model.fit(train[outcome_features], train["morte_evitavel_binaria"].astype(int))
    treatment_model.fit(train[treatment_features], train["binary_treatment"].astype(int))
    return outcome_model, treatment_model


def prepare_pair_data(
    df: pd.DataFrame,
    adjustment_set: list[str],
    reference_level: str,
    treated_level: str,
    min_cell_count: int,
    support_source_df: pd.DataFrame | None = None,
    support_keys: set[tuple[str, ...]] | None = None,
) -> tuple[pd.DataFrame, int]:
    pair = df[df[TREATMENT].isin([reference_level, treated_level])].copy()
    n_pair_before_support = len(pair)

    if support_keys is None:
        support_source = pair if support_source_df is None else support_source_df
        mask = common_support_mask(
            support_source,
            adjustment_set=adjustment_set,
            level_a=reference_level,
            level_b=treated_level,
            min_cell_count=min_cell_count,
        )
        support_keys = set(
            map(
                tuple,
                support_source.loc[mask, adjustment_set].astype(str).to_numpy(),
            )
        )

    pair_keys = list(map(tuple, pair[adjustment_set].astype(str).to_numpy()))
    pair = pair.loc[[key in support_keys for key in pair_keys]].copy()
    pair["binary_treatment"] = (pair[TREATMENT] == treated_level).astype(int)
    return pair, n_pair_before_support - len(pair)


def support_keys_for_pair(
    df: pd.DataFrame,
    adjustment_set: list[str],
    reference_level: str,
    treated_level: str,
    min_cell_count: int,
) -> set[tuple[str, ...]]:
    mask = common_support_mask(
        df,
        adjustment_set=adjustment_set,
        level_a=reference_level,
        level_b=treated_level,
        min_cell_count=min_cell_count,
    )
    return set(map(tuple, df.loc[mask, adjustment_set].astype(str).to_numpy()))


def build_support_key_map(
    df: pd.DataFrame,
    adjustment_set: list[str],
    min_cell_count: int,
) -> dict[str, set[tuple[str, ...]]]:
    return {
        f"{treated_level}_vs_{reference_level}": support_keys_for_pair(
            df,
            adjustment_set=adjustment_set,
            reference_level=reference_level,
            treated_level=treated_level,
            min_cell_count=min_cell_count,
        )
        for reference_level, treated_level in CONTRASTS
    }


def predict_outcome_under(
    outcome_model: Pipeline,
    eval_df: pd.DataFrame,
    adjustment_set: list[str],
    treatment_value: int,
) -> np.ndarray:
    features = model_features(include_treatment=True, adjustment_set=adjustment_set)
    counterfactual = eval_df[features].copy()
    counterfactual["binary_treatment"] = treatment_value
    return outcome_model.predict_proba(counterfactual[features])[:, 1]


def aipw_for_pair(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    adjustment_set: list[str],
    reference_level: str,
    treated_level: str,
    model_type: str,
    random_seed: int,
    min_cell_count: int,
    propensity_clip: float,
    support_source_df: pd.DataFrame | None = None,
    support_keys: set[tuple[str, ...]] | None = None,
) -> tuple[dict, dict]:
    train_pair, _ = prepare_pair_data(
        train_df,
        adjustment_set=adjustment_set,
        reference_level=reference_level,
        treated_level=treated_level,
        min_cell_count=min_cell_count,
        support_source_df=support_source_df,
        support_keys=support_keys,
    )
    eval_pair, n_removed_support = prepare_pair_data(
        eval_df,
        adjustment_set=adjustment_set,
        reference_level=reference_level,
        treated_level=treated_level,
        min_cell_count=min_cell_count,
        support_source_df=support_source_df,
        support_keys=support_keys,
    )

    contrast = f"{treated_level}_vs_{reference_level}"
    base_result = {
        "contrast": contrast,
        "treated_level": treated_level,
        "reference_level": reference_level,
        "n_used": len(eval_pair),
        "n_removed_no_common_support": n_removed_support,
        "risk_treated": np.nan,
        "risk_reference": np.nan,
        "risk_difference": np.nan,
        "risk_ratio": np.nan,
        "model_type": model_type,
        "support_type": f"pairwise_common_support_min_cell_{min_cell_count}",
    }

    if train_pair.empty or eval_pair.empty:
        return base_result, propensity_diagnostics(contrast, treated_level, reference_level, np.array([]), model_type, len(eval_pair), n_removed_support)

    if train_pair["binary_treatment"].nunique() < 2 or train_pair["morte_evitavel_binaria"].nunique() < 2:
        return base_result, propensity_diagnostics(contrast, treated_level, reference_level, np.array([]), model_type, len(eval_pair), n_removed_support)

    category_df = pd.concat(
        [train_pair, eval_pair],
        ignore_index=True,
        sort=False,
    )
    outcome_model, treatment_model = fit_nuisance_models(
        train_pair,
        category_df=category_df,
        adjustment_set=adjustment_set,
        model_type=model_type,
        random_seed=random_seed,
    )

    treatment_features = model_features(include_treatment=False, adjustment_set=adjustment_set)
    propensity = treatment_model.predict_proba(eval_pair[treatment_features])[:, 1]
    clipped_propensity = np.clip(propensity, propensity_clip, 1 - propensity_clip)

    mu1 = predict_outcome_under(outcome_model, eval_pair, adjustment_set, treatment_value=1)
    mu0 = predict_outcome_under(outcome_model, eval_pair, adjustment_set, treatment_value=0)
    treatment = eval_pair["binary_treatment"].to_numpy()
    outcome = eval_pair["morte_evitavel_binaria"].to_numpy()

    psi1 = mu1 + treatment * (outcome - mu1) / clipped_propensity
    psi0 = mu0 + (1 - treatment) * (outcome - mu0) / (1 - clipped_propensity)
    risk_treated = float(np.mean(psi1))
    risk_reference = float(np.mean(psi0))

    result = base_result | {
        "risk_treated": risk_treated,
        "risk_reference": risk_reference,
        "risk_difference": risk_treated - risk_reference,
        "risk_ratio": risk_treated / risk_reference if risk_reference > 0 else np.nan,
    }

    diagnostics = propensity_diagnostics(
        contrast,
        treated_level,
        reference_level,
        propensity,
        model_type,
        len(eval_pair),
        n_removed_support,
    )
    return result, diagnostics


def aipw_crossfit_for_pair(
    df: pd.DataFrame,
    adjustment_set: list[str],
    reference_level: str,
    treated_level: str,
    model_type: str,
    random_seed: int,
    min_cell_count: int,
    propensity_clip: float,
    crossfit_folds: int,
    support_source_df: pd.DataFrame | None = None,
    support_keys: set[tuple[str, ...]] | None = None,
) -> tuple[dict, dict]:
    eval_pair, n_removed_support = prepare_pair_data(
        df,
        adjustment_set=adjustment_set,
        reference_level=reference_level,
        treated_level=treated_level,
        min_cell_count=min_cell_count,
        support_source_df=support_source_df,
        support_keys=support_keys,
    )

    contrast = f"{treated_level}_vs_{reference_level}"
    base_result = {
        "contrast": contrast,
        "treated_level": treated_level,
        "reference_level": reference_level,
        "n_used": len(eval_pair),
        "n_removed_no_common_support": n_removed_support,
        "risk_treated": np.nan,
        "risk_reference": np.nan,
        "risk_difference": np.nan,
        "risk_ratio": np.nan,
        "model_type": model_type,
        "support_type": f"pairwise_common_support_min_cell_{min_cell_count}",
        "crossfit_folds": crossfit_folds,
    }

    if eval_pair.empty or eval_pair["binary_treatment"].nunique() < 2:
        return base_result, propensity_diagnostics(contrast, treated_level, reference_level, np.array([]), model_type, len(eval_pair), n_removed_support)
    if eval_pair["morte_evitavel_binaria"].nunique() < 2:
        return base_result, propensity_diagnostics(contrast, treated_level, reference_level, np.array([]), model_type, len(eval_pair), n_removed_support)

    fold_count = min(crossfit_folds, int(eval_pair["binary_treatment"].value_counts().min()))
    if fold_count < 2:
        return aipw_for_pair(
            train_df=df,
            eval_df=df,
            adjustment_set=adjustment_set,
            reference_level=reference_level,
            treated_level=treated_level,
            model_type=model_type,
            random_seed=random_seed,
            min_cell_count=min_cell_count,
            propensity_clip=propensity_clip,
            support_source_df=support_source_df,
            support_keys=support_keys,
        )

    splitter = StratifiedKFold(n_splits=fold_count, shuffle=True, random_state=random_seed)
    psi1 = np.empty(len(eval_pair), dtype=float)
    psi0 = np.empty(len(eval_pair), dtype=float)
    propensities = np.empty(len(eval_pair), dtype=float)
    outcome_features = model_features(include_treatment=True, adjustment_set=adjustment_set)
    treatment_features = model_features(include_treatment=False, adjustment_set=adjustment_set)
    y_strata = eval_pair["binary_treatment"].astype(int).to_numpy()

    for fold_number, (train_index, eval_index) in enumerate(splitter.split(eval_pair, y_strata), start=1):
        print(f"Cross-fit {contrast}: fold {fold_number}/{fold_count}.")
        train_fold = eval_pair.iloc[train_index].copy()
        eval_fold = eval_pair.iloc[eval_index].copy()
        if train_fold["binary_treatment"].nunique() < 2 or train_fold["morte_evitavel_binaria"].nunique() < 2:
            raise ValueError(f"Fold invalido para {contrast}: faltou variacao em tratamento ou desfecho.")

        outcome_model, treatment_model = fit_nuisance_models(
            train_fold,
            category_df=eval_pair,
            adjustment_set=adjustment_set,
            model_type=model_type,
            random_seed=random_seed,
        )
        propensity = treatment_model.predict_proba(eval_fold[treatment_features])[:, 1]
        clipped_propensity = np.clip(propensity, propensity_clip, 1 - propensity_clip)
        mu1 = predict_outcome_under(outcome_model, eval_fold, adjustment_set, treatment_value=1)
        mu0 = predict_outcome_under(outcome_model, eval_fold, adjustment_set, treatment_value=0)
        treatment = eval_fold["binary_treatment"].to_numpy()
        outcome = eval_fold["morte_evitavel_binaria"].to_numpy()

        psi1[eval_index] = mu1 + treatment * (outcome - mu1) / clipped_propensity
        psi0[eval_index] = mu0 + (1 - treatment) * (outcome - mu0) / (1 - clipped_propensity)
        propensities[eval_index] = propensity

    risk_treated = float(np.mean(psi1))
    risk_reference = float(np.mean(psi0))
    result = base_result | {
        "risk_treated": risk_treated,
        "risk_reference": risk_reference,
        "risk_difference": risk_treated - risk_reference,
        "risk_ratio": risk_treated / risk_reference if risk_reference > 0 else np.nan,
    }

    diagnostics = propensity_diagnostics(
        contrast,
        treated_level,
        reference_level,
        propensities,
        model_type,
        len(eval_pair),
        n_removed_support,
    )
    diagnostics["crossfit_folds"] = fold_count
    return result, diagnostics


def propensity_diagnostics(
    contrast: str,
    treated_level: str,
    reference_level: str,
    propensity: np.ndarray,
    model_type: str,
    n_used: int,
    n_removed_support: int,
) -> dict:
    if len(propensity) == 0:
        return {
            "contrast": contrast,
            "treated_level": treated_level,
            "reference_level": reference_level,
            "model_type": model_type,
            "n_used": n_used,
            "n_removed_no_common_support": n_removed_support,
            "propensity_mean": np.nan,
            "propensity_min": np.nan,
            "propensity_p01": np.nan,
            "propensity_p05": np.nan,
            "propensity_p50": np.nan,
            "propensity_p95": np.nan,
            "propensity_p99": np.nan,
            "propensity_max": np.nan,
            "propensity_extreme_share": np.nan,
        }

    return {
        "contrast": contrast,
        "treated_level": treated_level,
        "reference_level": reference_level,
        "model_type": model_type,
        "n_used": n_used,
        "n_removed_no_common_support": n_removed_support,
        "propensity_mean": float(np.mean(propensity)),
        "propensity_min": float(np.min(propensity)),
        "propensity_p01": float(np.quantile(propensity, 0.01)),
        "propensity_p05": float(np.quantile(propensity, 0.05)),
        "propensity_p50": float(np.quantile(propensity, 0.50)),
        "propensity_p95": float(np.quantile(propensity, 0.95)),
        "propensity_p99": float(np.quantile(propensity, 0.99)),
        "propensity_max": float(np.max(propensity)),
        "propensity_extreme_share": float(np.mean((propensity < 0.01) | (propensity > 0.99))),
    }


def estimate_aipw(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    adjustment_set: list[str],
    model_type: str,
    random_seed: int,
    min_cell_count: int,
    propensity_clip: float,
    support_source_df: pd.DataFrame | None = None,
    support_key_map: dict[str, set[tuple[str, ...]]] | None = None,
    crossfit_folds: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    effect_rows = []
    diagnostic_rows = []

    for reference_level, treated_level in CONTRASTS:
        contrast = f"{treated_level}_vs_{reference_level}"
        support_keys = None if support_key_map is None else support_key_map.get(contrast)
        if crossfit_folds > 1 and train_df is eval_df:
            effect, diagnostics = aipw_crossfit_for_pair(
                df=train_df,
                adjustment_set=adjustment_set,
                reference_level=reference_level,
                treated_level=treated_level,
                model_type=model_type,
                random_seed=random_seed,
                min_cell_count=min_cell_count,
                propensity_clip=propensity_clip,
                crossfit_folds=crossfit_folds,
                support_source_df=support_source_df,
                support_keys=support_keys,
            )
        else:
            effect, diagnostics = aipw_for_pair(
                train_df=train_df,
                eval_df=eval_df,
                adjustment_set=adjustment_set,
                reference_level=reference_level,
                treated_level=treated_level,
                model_type=model_type,
                random_seed=random_seed,
                min_cell_count=min_cell_count,
                propensity_clip=propensity_clip,
                support_source_df=support_source_df,
                support_keys=support_keys,
            )
        effect_rows.append(effect)
        diagnostic_rows.append(diagnostics)

    return pd.DataFrame(effect_rows), pd.DataFrame(diagnostic_rows)


def bootstrap_aipw(
    df: pd.DataFrame,
    adjustment_set: list[str],
    model_type: str,
    random_seed: int,
    min_cell_count: int,
    propensity_clip: float,
    n_iterations: int,
    bootstrap_sample_size: int | None,
    evaluation_sample_size: int | None,
    support_source_df: pd.DataFrame | None,
    support_key_map: dict[str, set[tuple[str, ...]]] | None,
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
        effects, _ = estimate_aipw(
            train_df=train_df,
            eval_df=eval_df,
            adjustment_set=adjustment_set,
            model_type=model_type,
            random_seed=seed,
            min_cell_count=min_cell_count,
            propensity_clip=propensity_clip,
            support_source_df=support_source_df,
            support_key_map=support_key_map,
        )
        effects.insert(0, "bootstrap_iteration", iteration)
        effects.insert(1, "bootstrap_sample_size", train_n)
        effects.insert(2, "bootstrap_evaluation_sample_size", eval_n)
        draw_rows.append(effects)

        if iteration == 1 or iteration % 10 == 0 or iteration == n_iterations:
            print(f"Bootstrap AIPW {iteration}/{n_iterations} concluido.")

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
                "model_type": model_type,
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
                "model_type": args.model_type,
                "outcome_filter": "mantem morte_evitavel em {0, 1}; exclui 2 (mal definida)",
                "treatment": TREATMENT,
                "treatment_levels": "|".join(TREATMENT_LEVELS),
                "adjustment_set": "|".join(adjustment_set),
                "n_rows": len(df),
                "outcome_mean": float(df["morte_evitavel_binaria"].mean()) if len(df) else np.nan,
                "min_cell_count": args.min_cell_count,
                "propensity_clip": args.propensity_clip,
            }
        ]
    )


def compare_with_gcomp(aipw: pd.DataFrame, gcomp_path: Path) -> pd.DataFrame:
    if not gcomp_path.exists():
        return pd.DataFrame()

    gcomp = pd.read_csv(gcomp_path)
    gcomp = gcomp.rename(
        columns={
            "risk_difference": "gcomp_risk_difference",
            "risk_ratio": "gcomp_risk_ratio",
        }
    )
    aipw_comp = aipw.rename(
        columns={
            "risk_difference": "aipw_risk_difference",
            "risk_ratio": "aipw_risk_ratio",
        }
    )
    merged = gcomp[["contrast", "gcomp_risk_difference", "gcomp_risk_ratio"]].merge(
        aipw_comp[["contrast", "aipw_risk_difference", "aipw_risk_ratio"]],
        on="contrast",
        how="inner",
    )
    return merged


def write_run_config(args: argparse.Namespace, adjustment_set: list[str]) -> None:
    config = {
        "dataset": str(args.dataset),
        "dag": str(args.dag),
        "output_dir": str(args.output_dir),
        "model_type": args.model_type,
        "max_rows": args.max_rows,
        "random_seed": args.random_seed,
        "min_cell_count": args.min_cell_count,
        "propensity_clip": args.propensity_clip,
        "bootstrap_iterations": args.bootstrap_iterations,
        "bootstrap_sample_size": args.bootstrap_sample_size,
        "bootstrap_evaluation_sample_size": args.bootstrap_evaluation_sample_size,
        "crossfit_folds": args.crossfit_folds,
        "adjustment_set": adjustment_set,
        "support_definition": "pairwise_common_support_keys_from_full_analysis_base",
        "evaluation_population": "observed_pair_levels_with_fixed_common_support",
        "treatment_encoding": "binary_per_pair_contrast",
    }
    with (args.output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estima contrastes de escolaridade via AIPW em suporte comum par-a-par."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dag", type=Path, default=DEFAULT_DAG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-type", choices=["logistic", "hgb", "xgb"], default="logistic")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-cell-count", type=int, default=20)
    parser.add_argument("--propensity-clip", type=float, default=0.01)
    parser.add_argument("--bootstrap-iterations", type=int, default=0)
    parser.add_argument("--bootstrap-sample-size", type=int, default=100000)
    parser.add_argument("--bootstrap-evaluation-sample-size", type=int, default=100000)
    parser.add_argument("--crossfit-folds", type=int, default=1)
    parser.add_argument("--gcomp-results", type=Path, default=GCOMP_RESULTS_PATH)
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
    support_key_map = build_support_key_map(
        df,
        adjustment_set=adjustment_set,
        min_cell_count=args.min_cell_count,
    )

    effects, diagnostics = estimate_aipw(
        train_df=df,
        eval_df=df,
        adjustment_set=adjustment_set,
        model_type=args.model_type,
        random_seed=args.random_seed,
        min_cell_count=args.min_cell_count,
        propensity_clip=args.propensity_clip,
        support_source_df=df,
        support_key_map=support_key_map,
        crossfit_folds=args.crossfit_folds,
    )

    bootstrap_sample_size = None if args.bootstrap_sample_size == 0 else args.bootstrap_sample_size
    evaluation_sample_size = None if args.bootstrap_evaluation_sample_size == 0 else args.bootstrap_evaluation_sample_size
    bootstrap_draws, bootstrap_summary = bootstrap_aipw(
        df=df,
        adjustment_set=adjustment_set,
        model_type=args.model_type,
        random_seed=args.random_seed,
        min_cell_count=args.min_cell_count,
        propensity_clip=args.propensity_clip,
        n_iterations=args.bootstrap_iterations,
        bootstrap_sample_size=bootstrap_sample_size,
        evaluation_sample_size=evaluation_sample_size,
        support_source_df=df,
        support_key_map=support_key_map,
    )
    sample_summary(df, adjustment_set, args).to_csv(args.output_dir / "aipw_sample_summary.csv", index=False)
    effects.to_csv(args.output_dir / "aipw_effect_estimates_common_support.csv", index=False)
    diagnostics.to_csv(args.output_dir / "aipw_propensity_diagnostics.csv", index=False)

    if not bootstrap_draws.empty:
        bootstrap_draws.to_csv(
            args.output_dir / "aipw_bootstrap_effect_estimates_common_support.csv",
            index=False,
        )
        bootstrap_summary.to_csv(
            args.output_dir / "aipw_bootstrap_summary_common_support.csv",
            index=False,
        )

    comparison = compare_with_gcomp(effects, args.gcomp_results)
    if not comparison.empty:
        comparison.to_csv(args.output_dir / "aipw_vs_gcomp_comparison.csv", index=False)

    write_run_config(args, adjustment_set)

    print("Estimacao AIPW concluida.")
    print(f"Linhas na analise: {len(df):,}")
    print(f"Modelo: {args.model_type}")
    print(f"Ajuste: {', '.join(adjustment_set)}")
    print(f"Saidas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
