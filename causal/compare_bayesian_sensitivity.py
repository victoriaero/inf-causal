from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "education_effect_bayesian_sensitivity"
DEFAULT_RUNS = {
    "conservative": PROJECT_ROOT / "causal" / "output" / "education_effect_bayesian_hierarchical_conservative_svi3000",
    "default": PROJECT_ROOT / "causal" / "output" / "education_effect_bayesian_hierarchical_default_svi3000",
    "weak": PROJECT_ROOT / "causal" / "output" / "education_effect_bayesian_hierarchical_weak_svi3000",
}


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use o formato nome=/caminho/da/pasta")
    name, path = value.split("=", 1)
    return name.strip(), Path(path)


def read_run(name: str, path: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    effects_path = path / "bayesian_posterior_effects_summary.csv"
    risks_path = path / "bayesian_posterior_risks_summary.csv"
    predictive_path = path / "bayesian_posterior_predictive_by_treatment.csv"

    for file_path in [effects_path, risks_path, predictive_path]:
        if not file_path.exists():
            raise FileNotFoundError(file_path)

    effects = pd.read_csv(effects_path)
    effects.insert(0, "run", name)
    risks = pd.read_csv(risks_path)
    risks.insert(0, "run", name)
    predictive = pd.read_csv(predictive_path)
    predictive.insert(0, "run", name)
    return effects, risks, predictive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compara sensibilidades dos priors bayesianos.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run", action="append", type=parse_run, default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = DEFAULT_RUNS.copy()
    for name, path in args.run:
        runs[name] = path

    effects_rows = []
    risk_rows = []
    predictive_rows = []
    for name, path in runs.items():
        effects, risks, predictive = read_run(name, path)
        effects_rows.append(effects)
        risk_rows.append(risks)
        predictive_rows.append(predictive)

    effects = pd.concat(effects_rows, ignore_index=True)
    risks = pd.concat(risk_rows, ignore_index=True)
    predictive = pd.concat(predictive_rows, ignore_index=True)

    effects.to_csv(args.output_dir / "bayesian_sensitivity_effects_long.csv", index=False)
    risks.to_csv(args.output_dir / "bayesian_sensitivity_risks_long.csv", index=False)
    predictive.to_csv(args.output_dir / "bayesian_sensitivity_predictive_long.csv", index=False)

    effects.pivot(index="contrast", columns="run", values="risk_difference_median").to_csv(
        args.output_dir / "bayesian_sensitivity_risk_difference_median.csv"
    )
    effects.pivot(index="contrast", columns="run", values="risk_ratio_median").to_csv(
        args.output_dir / "bayesian_sensitivity_risk_ratio_median.csv"
    )
    risks.pivot(index="level", columns="run", values="median").to_csv(
        args.output_dir / "bayesian_sensitivity_risk_median.csv"
    )

    print("Comparacao bayesiana concluida.")
    print(f"Rodadas: {', '.join(runs)}")
    print(f"Saidas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
