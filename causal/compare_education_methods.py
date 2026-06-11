from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "education_effect_comparison"
DEFAULT_METHOD_PATHS = {
    "gcomp": PROJECT_ROOT / "causal" / "output" / "education_effect" / "effect_estimates_common_support.csv",
    "aipw_logistic": PROJECT_ROOT / "causal" / "output" / "education_effect_aipw" / "aipw_effect_estimates_common_support.csv",
    "aipw_hgb": PROJECT_ROOT / "causal" / "output" / "education_effect_aipw_hgb" / "aipw_effect_estimates_common_support.csv",
    "aipw_xgb": PROJECT_ROOT / "causal" / "output" / "education_effect_aipw_xgb" / "aipw_effect_estimates_common_support.csv",
    "aipw_xgb_pair_crossfit3_bootstrap300": PROJECT_ROOT
    / "causal"
    / "output"
    / "education_effect_aipw_xgb_crossfit3_bootstrap_300x100k"
    / "aipw_bootstrap_summary_common_support.csv",
    "aipw_xgb_3class_bootstrap300": PROJECT_ROOT
    / "causal"
    / "output"
    / "education_effect_aipw_xgb_3class_bootstrap_300x100k"
    / "aipw_3class_bootstrap_summary_global_support.csv",
}


def read_method_results(method: str, path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Resultado nao encontrado para {method}: {path}")

    df = pd.read_csv(path)
    if method == "gcomp":
        rename_columns = {
            "comparison_level": "treated_level",
            "risk_comparison": "risk_treated",
        }
        if "n_used" not in df.columns:
            rename_columns["n_rows_standardized_over"] = "n_used"
        df = df.rename(columns=rename_columns)
        if "n_removed_no_common_support" not in df.columns:
            df["n_removed_no_common_support"] = pd.NA
    else:
        df = df.copy()

    rename_columns = {}
    if "estimate_risk_difference" in df.columns and "risk_difference" not in df.columns:
        rename_columns["estimate_risk_difference"] = "risk_difference"
    if "estimate_risk_ratio" in df.columns and "risk_ratio" not in df.columns:
        rename_columns["estimate_risk_ratio"] = "risk_ratio"
    df = df.rename(columns=rename_columns)

    for optional in ["n_used", "n_removed_no_common_support", "risk_treated", "risk_reference"]:
        if optional not in df.columns:
            df[optional] = pd.NA

    required = [
        "contrast",
        "treated_level",
        "reference_level",
        "n_used",
        "n_removed_no_common_support",
        "risk_treated",
        "risk_reference",
        "risk_difference",
        "risk_ratio",
    ]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colunas ausentes em {path}: {missing}")

    out = df[required].copy()
    out.insert(0, "method", method)
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
        description="Compara estimativas de escolaridade em formato padronizado."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--include-defaults",
        action="store_true",
        help="Inclui os caminhos padrao alem dos resultados informados por --method-result.",
    )
    parser.add_argument(
        "--method-result",
        action="append",
        type=parse_method_path,
        default=[],
        help="Resultado extra ou substituto no formato metodo=/caminho/arquivo.csv.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    method_paths = DEFAULT_METHOD_PATHS.copy() if args.include_defaults or not args.method_result else {}
    for method, path in args.method_result:
        method_paths[method] = path

    result_frames = [read_method_results(method, path) for method, path in method_paths.items()]
    combined = pd.concat(
        [frame.dropna(axis=1, how="all") for frame in result_frames],
        ignore_index=True,
        sort=False,
    )
    combined = combined.sort_values(["contrast", "method"]).reset_index(drop=True)

    rd_wide = combined.pivot(index="contrast", columns="method", values="risk_difference")
    rr_wide = combined.pivot(index="contrast", columns="method", values="risk_ratio")
    n_wide = combined.pivot(index="contrast", columns="method", values="n_used")

    combined.to_csv(args.output_dir / "education_effect_methods_long.csv", index=False)
    rd_wide.to_csv(args.output_dir / "education_effect_methods_risk_difference.csv")
    rr_wide.to_csv(args.output_dir / "education_effect_methods_risk_ratio.csv")
    n_wide.to_csv(args.output_dir / "education_effect_methods_n_used.csv")

    print("Comparacao concluida.")
    print(f"Metodos: {', '.join(method_paths)}")
    print(f"Saidas salvas em: {args.output_dir}")


if __name__ == "__main__":
    main()
