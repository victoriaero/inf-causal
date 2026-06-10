from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    import pyro
    import pyro.distributions as dist
    from pyro.infer import SVI, Trace_ELBO
    from pyro.infer.autoguide import AutoNormal
    from pyro.optim import ClippedAdam
except ImportError:  # pragma: no cover - handled with a clear runtime message.
    pyro = None
    dist = None
    SVI = None
    Trace_ELBO = None
    AutoNormal = None
    ClippedAdam = None

from estimate_education_effect import (
    CONTRASTS,
    OUTCOME,
    PROJECT_ROOT,
    TREATMENT,
    TREATMENT_LEVELS,
    prepare_analysis_data,
)
from estimate_education_effect_aipw_xgb_3class import (
    apply_global_support,
    global_support_keys,
    global_support_summary,
)
from test_final_dag import DEFAULT_DAG_PATH, DEFAULT_DATASET_PATH, build_graph, load_dag_config


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "education_effect_bayesian_hierarchical"


def require_pyro() -> None:
    if pyro is None:
        raise ImportError(
            "pyro-ppl nao esta instalado neste ambiente. "
            "Instale com: python3 -m pip install pyro-ppl"
        )


def logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1 - 1e-6)
    return float(np.log(clipped / (1 - clipped)))


def encode_categories(
    df: pd.DataFrame,
    adjustment_set: list[str],
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    encoded = df.copy()
    levels = {TREATMENT: TREATMENT_LEVELS}
    encoded["treatment_code"] = pd.Categorical(
        encoded[TREATMENT],
        categories=TREATMENT_LEVELS,
        ordered=True,
    ).codes

    for column in adjustment_set:
        column_levels = sorted(encoded[column].dropna().astype(str).unique().tolist())
        levels[column] = column_levels
        encoded[f"{column}_code"] = pd.Categorical(
            encoded[column].astype(str),
            categories=column_levels,
            ordered=False,
        ).codes

    return encoded, levels


def aggregate_training_cells(
    df: pd.DataFrame,
    adjustment_set: list[str],
) -> pd.DataFrame:
    group_cols = [TREATMENT] + adjustment_set
    cells = (
        df.groupby(group_cols, dropna=False, observed=True)["morte_evitavel_binaria"]
        .agg(successes="sum", total="size")
        .reset_index()
    )
    return cells


def aggregate_evaluation_strata(
    df: pd.DataFrame,
    adjustment_set: list[str],
) -> pd.DataFrame:
    return (
        df.groupby(adjustment_set, dropna=False, observed=True)
        .size()
        .reset_index(name="total")
    )


def tensors_from_cells(
    cells: pd.DataFrame,
    adjustment_set: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors = {
        "treatment": torch.as_tensor(cells["treatment_code"].to_numpy(), dtype=torch.long, device=device),
        "successes": torch.as_tensor(cells["successes"].to_numpy(), dtype=torch.float32, device=device),
        "total": torch.as_tensor(cells["total"].to_numpy(), dtype=torch.float32, device=device),
    }
    for column in adjustment_set:
        tensors[column] = torch.as_tensor(cells[f"{column}_code"].to_numpy(), dtype=torch.long, device=device)
    return tensors


def tensors_from_eval_strata(
    strata: pd.DataFrame,
    adjustment_set: list[str],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    tensors = {
        "weights": torch.as_tensor(strata["total"].to_numpy(), dtype=torch.float32, device=device),
    }
    for column in adjustment_set:
        tensors[column] = torch.as_tensor(strata[f"{column}_code"].to_numpy(), dtype=torch.long, device=device)
    return tensors


def hierarchical_binomial_model(
    treatment: torch.Tensor,
    covariates: dict[str, torch.Tensor],
    total: torch.Tensor,
    successes: torch.Tensor | None,
    n_levels: dict[str, int],
    adjustment_set: list[str],
    intercept_loc: float,
) -> None:
    intercept = pyro.sample("intercept", dist.Normal(intercept_loc, 1.5))
    treatment_effect = pyro.sample(
        "treatment_effect",
        dist.Normal(0.0, 1.0).expand([len(TREATMENT_LEVELS) - 1]).to_event(1),
    )
    treatment_effect = torch.cat(
        [torch.zeros(1, device=treatment.device), treatment_effect],
        dim=0,
    )

    logits = intercept + treatment_effect[treatment]

    for column in adjustment_set:
        sigma = pyro.sample(f"sigma_{column}", dist.HalfNormal(1.0))
        raw_effect = pyro.sample(
            f"effect_{column}",
            dist.Normal(0.0, sigma).expand([n_levels[column]]).to_event(1),
        )
        centered_effect = raw_effect - raw_effect.mean()
        logits = logits + centered_effect[covariates[column]]

    with pyro.plate("cells", len(total)):
        pyro.sample(
            "obs",
            dist.Binomial(total_count=total, logits=logits),
            obs=successes,
        )


def fit_svi(
    tensors: dict[str, torch.Tensor],
    n_levels: dict[str, int],
    adjustment_set: list[str],
    intercept_loc: float,
    svi_steps: int,
    learning_rate: float,
    random_seed: int,
) -> tuple[AutoNormal, pd.DataFrame]:
    pyro.clear_param_store()
    pyro.set_rng_seed(random_seed)

    def model() -> None:
        covariates = {column: tensors[column] for column in adjustment_set}
        hierarchical_binomial_model(
            treatment=tensors["treatment"],
            covariates=covariates,
            total=tensors["total"],
            successes=tensors["successes"],
            n_levels=n_levels,
            adjustment_set=adjustment_set,
            intercept_loc=intercept_loc,
        )

    guide = AutoNormal(model)
    optimizer = ClippedAdam({"lr": learning_rate, "clip_norm": 10.0})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rows = []
    for step in range(1, svi_steps + 1):
        loss = float(svi.step())
        if step == 1 or step % 100 == 0 or step == svi_steps:
            avg_loss = loss / float(tensors["total"].sum().detach().cpu())
            print(f"SVI {step}/{svi_steps}: loss_per_row={avg_loss:.6f}")
            rows.append({"step": step, "loss": loss, "loss_per_row": avg_loss})

    return guide, pd.DataFrame(rows)


def sample_posterior_parameters(
    guide: AutoNormal,
    n_samples: int,
) -> dict[str, torch.Tensor]:
    samples: dict[str, list[torch.Tensor]] = {}
    for _ in range(n_samples):
        draw = guide()
        for name, value in draw.items():
            samples.setdefault(name, []).append(value.detach())
    return {name: torch.stack(values, dim=0) for name, values in samples.items()}


def posterior_standardized_risks(
    samples: dict[str, torch.Tensor],
    eval_tensors: dict[str, torch.Tensor],
    n_levels: dict[str, int],
    adjustment_set: list[str],
) -> pd.DataFrame:
    weights = eval_tensors["weights"]
    weight_sum = weights.sum()
    rows = []

    for sample_index in range(samples["intercept"].shape[0]):
        intercept = samples["intercept"][sample_index]
        treatment_effect = torch.cat(
            [
                torch.zeros(1, device=weights.device),
                samples["treatment_effect"][sample_index],
            ],
            dim=0,
        )
        base_logits = intercept
        for column in adjustment_set:
            raw = samples[f"effect_{column}"][sample_index]
            centered = raw - raw.mean()
            base_logits = base_logits + centered[eval_tensors[column]]

        for level_index, level in enumerate(TREATMENT_LEVELS):
            probabilities = torch.sigmoid(base_logits + treatment_effect[level_index])
            risk = torch.sum(probabilities * weights) / weight_sum
            rows.append(
                {
                    "posterior_sample": sample_index + 1,
                    "level": level,
                    "risk": float(risk.detach().cpu()),
                }
            )

    return pd.DataFrame(rows)


def summarize_posterior(risks: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    risk_summary = (
        risks.groupby("level", dropna=False)["risk"]
        .agg(
            mean="mean",
            median="median",
            ci95_low=lambda value: value.quantile(0.025),
            ci95_high=lambda value: value.quantile(0.975),
        )
        .reset_index()
    )

    wide = risks.pivot(index="posterior_sample", columns="level", values="risk").reset_index()
    rows = []
    for reference_level, comparison_level in CONTRASTS:
        risk_reference = wide[reference_level]
        risk_comparison = wide[comparison_level]
        risk_difference = risk_comparison - risk_reference
        risk_ratio = risk_comparison / risk_reference
        rows.append(
            {
                "contrast": f"{comparison_level}_vs_{reference_level}",
                "treated_level": comparison_level,
                "reference_level": reference_level,
                "risk_difference_mean": float(risk_difference.mean()),
                "risk_difference_median": float(risk_difference.median()),
                "risk_difference_ci95_low": float(risk_difference.quantile(0.025)),
                "risk_difference_ci95_high": float(risk_difference.quantile(0.975)),
                "risk_ratio_mean": float(risk_ratio.mean()),
                "risk_ratio_median": float(risk_ratio.median()),
                "risk_ratio_ci95_low": float(risk_ratio.quantile(0.025)),
                "risk_ratio_ci95_high": float(risk_ratio.quantile(0.975)),
            }
        )

    return risk_summary, pd.DataFrame(rows)


def write_run_config(args: argparse.Namespace, adjustment_set: list[str], output_dir: Path) -> None:
    config = {
        "dataset": str(args.dataset),
        "dag": str(args.dag),
        "output_dir": str(output_dir),
        "model_type": "bayesian_hierarchical_binomial_logistic",
        "inference": "pyro_svi_autonormal",
        "max_rows": args.max_rows,
        "random_seed": args.random_seed,
        "min_cell_count": args.min_cell_count,
        "svi_steps": args.svi_steps,
        "learning_rate": args.learning_rate,
        "posterior_samples": args.posterior_samples,
        "adjustment_set": adjustment_set,
        "support_definition": "global_three_level_common_support_keys_from_full_analysis_base",
        "evaluation_population": "adjustment_strata_with_global_common_support",
        "likelihood": "aggregated_binomial_by_treatment_and_adjustment_strata",
    }
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Modelo bayesiano hierarquico agregado para escolaridade e morte evitavel."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--dag", type=Path, default=DEFAULT_DAG_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-cell-count", type=int, default=20)
    parser.add_argument("--svi-steps", type=int, default=3000)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--posterior-samples", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    require_pyro()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo PyTorch: {device}")

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
    supported_df, n_removed_support = apply_global_support(df, adjustment_set, support_keys)
    print(f"Linhas na analise: {len(df):,}")
    print(f"Linhas no suporte global: {len(supported_df):,}")

    training_cells = aggregate_training_cells(supported_df, adjustment_set)
    eval_strata = aggregate_evaluation_strata(supported_df, adjustment_set)
    training_cells, levels = encode_categories(training_cells, adjustment_set)
    eval_strata, _ = encode_categories(eval_strata.assign(**{TREATMENT: TREATMENT_LEVELS[0]}), adjustment_set)
    n_levels = {column: len(levels[column]) for column in adjustment_set}

    tensors = tensors_from_cells(training_cells, adjustment_set, device)
    eval_tensors = tensors_from_eval_strata(eval_strata, adjustment_set, device)
    intercept_loc = logit(float(supported_df["morte_evitavel_binaria"].mean()))

    guide, loss_trace = fit_svi(
        tensors=tensors,
        n_levels=n_levels,
        adjustment_set=adjustment_set,
        intercept_loc=intercept_loc,
        svi_steps=args.svi_steps,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )

    posterior_samples = sample_posterior_parameters(guide, args.posterior_samples)
    posterior_risks = posterior_standardized_risks(
        posterior_samples,
        eval_tensors=eval_tensors,
        n_levels=n_levels,
        adjustment_set=adjustment_set,
    )
    risk_summary, effect_summary = summarize_posterior(posterior_risks)

    support_summary.to_csv(args.output_dir / "bayesian_global_support_summary.csv", index=False)
    support_treatment_distribution.to_csv(
        args.output_dir / "bayesian_global_support_treatment_distribution.csv",
        index=False,
    )
    training_cells.to_csv(args.output_dir / "bayesian_training_cells.csv", index=False)
    eval_strata.to_csv(args.output_dir / "bayesian_evaluation_strata.csv", index=False)
    loss_trace.to_csv(args.output_dir / "bayesian_svi_loss_trace.csv", index=False)
    posterior_risks.to_csv(args.output_dir / "bayesian_posterior_risks_draws.csv", index=False)
    risk_summary.to_csv(args.output_dir / "bayesian_posterior_risks_summary.csv", index=False)
    effect_summary.to_csv(args.output_dir / "bayesian_posterior_effects_summary.csv", index=False)

    metadata = pd.DataFrame(
        [
            {
                "dataset_path": str(args.dataset),
                "dag_path": str(args.dag),
                "n_rows_before_support": len(df),
                "n_rows_after_support": len(supported_df),
                "n_removed_no_common_support": n_removed_support,
                "n_training_cells": len(training_cells),
                "n_evaluation_strata": len(eval_strata),
                "outcome_mean_after_support": float(supported_df["morte_evitavel_binaria"].mean()),
                "adjustment_set": "|".join(adjustment_set),
            }
        ]
    )
    metadata.to_csv(args.output_dir / "bayesian_sample_summary.csv", index=False)
    write_run_config(args, adjustment_set, args.output_dir)

    print("Estimacao bayesiana concluida.")
    print(f"Celulas agregadas de treino: {len(training_cells):,}")
    print(f"Estratos de avaliacao: {len(eval_strata):,}")
    print(f"Saidas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
