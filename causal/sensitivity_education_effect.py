from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "education_effect_sensitivity"
DEFAULT_RESULT_PATHS = {
    "gcomp": PROJECT_ROOT / "causal" / "output" / "education_effect" / "effect_estimates_common_support.csv",
    "aipw_xgb_pair_crossfit3_bootstrap300": PROJECT_ROOT
    / "causal"
    / "output"
    / "education_effect_aipw_xgb_crossfit3_bootstrap_300x100k"
    / "aipw_bootstrap_summary_common_support.csv",
    "aipw_xgb_3class": PROJECT_ROOT
    / "causal"
    / "output"
    / "education_effect_aipw_xgb_3class_bootstrap_300x100k"
    / "aipw_3class_bootstrap_summary_global_support.csv",
    "bayesian_default": PROJECT_ROOT
    / "causal"
    / "output"
    / "education_effect_bayesian_hierarchical_default_svi3000"
    / "bayesian_posterior_effects_summary.csv",
    "bayesian_conservative": PROJECT_ROOT
    / "causal"
    / "output"
    / "education_effect_bayesian_hierarchical_conservative_svi3000"
    / "bayesian_posterior_effects_summary.csv",
    "bayesian_weak": PROJECT_ROOT
    / "causal"
    / "output"
    / "education_effect_bayesian_hierarchical_weak_svi3000"
    / "bayesian_posterior_effects_summary.csv",
}


def evalue_from_risk_ratio(risk_ratio: float) -> float:
    if pd.isna(risk_ratio) or risk_ratio <= 0:
        return np.nan

    rr_away_from_null = risk_ratio if risk_ratio >= 1 else 1 / risk_ratio
    if rr_away_from_null <= 1:
        return 1.0

    return float(rr_away_from_null + np.sqrt(rr_away_from_null * (rr_away_from_null - 1)))


def evalue_limit_from_ci(risk_ratio: float, ci_low: float, ci_high: float) -> float:
    if pd.isna(risk_ratio) or pd.isna(ci_low) or pd.isna(ci_high):
        return np.nan

    if ci_low <= 1 <= ci_high:
        return 1.0

    closest_to_null = ci_low if risk_ratio > 1 else ci_high
    return evalue_from_risk_ratio(closest_to_null)


def normalize_result(method: str, path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    normalized = df.copy()

    rename = {}
    if "comparison_level" in normalized.columns and "treated_level" not in normalized.columns:
        rename["comparison_level"] = "treated_level"
    if "risk_ratio_median" in normalized.columns and "risk_ratio" not in normalized.columns:
        rename["risk_ratio_median"] = "risk_ratio"
    if "estimate_risk_ratio" in normalized.columns and "risk_ratio" not in normalized.columns:
        rename["estimate_risk_ratio"] = "risk_ratio"
    if "estimate_risk_difference" in normalized.columns and "risk_difference" not in normalized.columns:
        rename["estimate_risk_difference"] = "risk_difference"
    if "risk_ratio_ci95_low" in normalized.columns and "ci95_low_risk_ratio" not in normalized.columns:
        rename["risk_ratio_ci95_low"] = "ci95_low_risk_ratio"
    if "risk_ratio_ci95_high" in normalized.columns and "ci95_high_risk_ratio" not in normalized.columns:
        rename["risk_ratio_ci95_high"] = "ci95_high_risk_ratio"
    normalized = normalized.rename(columns=rename)

    required = ["contrast", "treated_level", "reference_level", "risk_ratio"]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {missing}")

    keep = required.copy()
    for optional in [
        "risk_difference",
        "n_used",
        "n_removed_no_common_support",
        "ci95_low_risk_ratio",
        "ci95_high_risk_ratio",
        "model_type",
        "support_type",
    ]:
        if optional in normalized.columns:
            keep.append(optional)

    out = normalized[keep].copy()
    out.insert(0, "method", method)
    return out


def sensitivity_summary(
    results: pd.DataFrame,
    benchmark_rr: float | None = None,
) -> pd.DataFrame:
    out = results.copy()
    out["risk_ratio"] = pd.to_numeric(out["risk_ratio"], errors="coerce")
    out["risk_ratio_away_from_null"] = np.where(
        out["risk_ratio"] >= 1,
        out["risk_ratio"],
        1 / out["risk_ratio"],
    )
    out["e_value_point_estimate"] = out["risk_ratio"].map(evalue_from_risk_ratio)

    has_ci = {"ci95_low_risk_ratio", "ci95_high_risk_ratio"}.issubset(out.columns)
    if has_ci:
        out["e_value_ci_limit"] = [
            evalue_limit_from_ci(rr, low, high)
            for rr, low, high in zip(
                out["risk_ratio"],
                out["ci95_low_risk_ratio"],
                out["ci95_high_risk_ratio"],
            )
        ]
    else:
        out["e_value_ci_limit"] = np.nan

    if benchmark_rr is not None and benchmark_rr > 1:
        out["benchmark_rr"] = benchmark_rr
        out["e_value_exceeds_benchmark"] = out["e_value_point_estimate"] > benchmark_rr
        if has_ci:
            out["e_value_ci_limit_exceeds_benchmark"] = out["e_value_ci_limit"] > benchmark_rr

    return out


def parse_method_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Use o formato metodo=/caminho/resultado.csv")
    method, path = value.split("=", 1)
    method = method.strip()
    if not method:
        raise argparse.ArgumentTypeError("Nome do metodo vazio.")
    return method, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sensitivity check por E-value para o efeito de escolaridade sobre morte evitavel."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--result",
        action="append",
        type=parse_method_path,
        default=[],
        help="Resultado no formato metodo=/caminho/arquivo.csv. Se omitido, usa os caminhos padrao existentes.",
    )
    parser.add_argument(
        "--require-all-defaults",
        action="store_true",
        help="Falha se algum resultado padrao nao existir. Por padrao, usa apenas os arquivos existentes.",
    )
    parser.add_argument(
        "--benchmark-rr",
        type=float,
        default=0.0,
        help=(
            "RR do confundidor observado mais forte, usado como referencia para o E-value. "
            "Se informado (>1), adiciona colunas benchmark_rr e e_value_exceeds_benchmark. "
            "Exemplo: --benchmark-rr 1.4 se raca_cor tem RR ~1.4 com morte_evitavel."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method_paths = dict(args.result) if args.result else DEFAULT_RESULT_PATHS
    frames = []
    missing_paths = []

    for method, path in method_paths.items():
        if not path.exists():
            missing_paths.append((method, path))
            if args.result or args.require_all_defaults:
                raise FileNotFoundError(f"Resultado nao encontrado para {method}: {path}")
            continue
        frames.append(normalize_result(method, path))

    if not frames:
        missing_text = "\n".join(f"- {method}: {path}" for method, path in missing_paths)
        raise FileNotFoundError("Nenhum arquivo de resultado encontrado.\n" + missing_text)

    results = pd.concat(frames, ignore_index=True)
    benchmark_rr = args.benchmark_rr if args.benchmark_rr > 1 else None
    summary = sensitivity_summary(results, benchmark_rr=benchmark_rr)
    summary = summary.sort_values(["contrast", "method"]).reset_index(drop=True)
    summary.to_csv(args.output_dir / "education_effect_evalues.csv", index=False)

    run_summary = pd.DataFrame(
        [
            {
                "n_methods_loaded": len(frames),
                "methods_loaded": "|".join(summary["method"].drop_duplicates()),
                "n_missing_default_paths": len(missing_paths),
                "missing_default_paths": "|".join(f"{method}:{path}" for method, path in missing_paths),
                "benchmark_rr": benchmark_rr if benchmark_rr is not None else "",
                "metric": "E-value on risk-ratio scale",
                "note": (
                    "E-value resume a forca minima de associacao que um confundidor nao observado "
                    "precisaria ter com tratamento e desfecho, na escala RR, para explicar o efeito observado. "
                    "Use --benchmark-rr para comparar com o RR do confundidor observado mais forte."
                ),
            }
        ]
    )
    run_summary.to_csv(args.output_dir / "sensitivity_run_summary.csv", index=False)

    print("Sensitivity check concluido.")
    print(f"Metodos carregados: {', '.join(summary['method'].drop_duplicates())}")
    print(f"Saidas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
