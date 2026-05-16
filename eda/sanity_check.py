from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from plot_style import clean_axis, save_figure, set_plot_style

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "sim_selected" / "dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eda" / "output"

MISSING_LIKE = {
    "",
    " ",
    "nan",
    "na",
    "n/a",
    "none",
    "null",
    "<na>",
    "ignorado",
    "ignorada",
    "ign",
    "ign.",
    "desconhecido",
    "desconhecida",
    "não informado",
    "nao informado",
    "não informada",
    "nao informada",
    "sem informação",
    "sem informacao",
}

MISSING_CODES = {"9", "9.0", "99", "99.0", "999", "999.0", "9999", "9999.0"}
ID_OR_NUMERIC_COLUMNS = {
    "ano",
    "idade",
    "sequencial_obito",
    "id_municipio_residencia",
    "id_municipio_ocorrencia",
}


def read_dataset(path: Path) -> pd.DataFrame:
    code_columns = {
        "sequencial_obito",
        "causa_basica",
        "hora_obito",
        "naturalidade",
        "escolaridade",
        "sexo",
        "raca_cor",
        "estado_civil",
        "id_municipio_residencia",
        "id_municipio_ocorrencia",
        "ocupacao",
        "local_ocorrencia",
        "assistencia_medica",
        "necropsia",
        "codigo_estabelecimento",
        "atestante",
        "escolaridade_grupo",
    }
    dtype = {column: "string" for column in code_columns}
    df = pd.read_csv(path, dtype=dtype, low_memory=False)

    for column in ["ano", "idade", "morte_evitavel", "escolaridade_nivel"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def ignored_or_missing(series: pd.Series, column: str) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    text_missing = series.isna() | normalized.isin(MISSING_LIKE)

    if column in ID_OR_NUMERIC_COLUMNS:
        code_missing = pd.Series(False, index=series.index)
    else:
        code_missing = normalized.isin(MISSING_CODES)

    return text_missing | code_missing


def build_column_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_rows = len(df)

    for column in df.columns:
        missing_mask = ignored_or_missing(df[column], column)
        rows.append(
            {
                "column": column,
                "dtype": str(df[column].dtype),
                "n_rows": n_rows,
                "n_missing_or_ignored": int(missing_mask.sum()),
                "pct_missing_or_ignored": float(missing_mask.mean()) if n_rows else 0.0,
                "n_unique_non_missing": int(df.loc[~missing_mask, column].nunique(dropna=True)),
            }
        )

    return pd.DataFrame(rows).sort_values("pct_missing_or_ignored", ascending=False)


def build_dataset_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "n_rows": len(df),
        "n_columns": len(df.columns),
        "min_year": df["ano"].min() if "ano" in df.columns else pd.NA,
        "max_year": df["ano"].max() if "ano" in df.columns else pd.NA,
        "min_age": df["idade"].min() if "idade" in df.columns else pd.NA,
        "max_age": df["idade"].max() if "idade" in df.columns else pd.NA,
        "n_states": df["sigla_uf"].nunique(dropna=True) if "sigla_uf" in df.columns else pd.NA,
    }
    return pd.DataFrame([summary])


def build_year_state_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["ano", "sigla_uf"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["ano", "sigla_uf"])
    )


def build_death_date_checks(df: pd.DataFrame) -> pd.DataFrame:
    if "data_obito" not in df.columns or "ano" not in df.columns:
        return pd.DataFrame()

    dates = pd.to_datetime(df["data_obito"], errors="coerce")
    date_year = dates.dt.year
    mismatch = dates.notna() & df["ano"].notna() & (date_year != pd.to_numeric(df["ano"], errors="coerce"))

    return pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_invalid_death_dates": int(dates.isna().sum()),
                "pct_invalid_death_dates": float(dates.isna().mean()) if len(df) else 0.0,
                "n_year_mismatch": int(mismatch.sum()),
                "pct_year_mismatch": float(mismatch.mean()) if len(df) else 0.0,
            }
        ]
    )


def build_duplicate_checks(df: pd.DataFrame) -> pd.DataFrame:
    candidate_keys = ["ano", "sigla_uf", "sequencial_obito"]
    present_keys = [column for column in candidate_keys if column in df.columns]

    if len(present_keys) != len(candidate_keys):
        return pd.DataFrame()

    valid_key_rows = df[present_keys].notna().all(axis=1)
    duplicates = df.loc[valid_key_rows, present_keys].duplicated(keep=False)

    return pd.DataFrame(
        [
            {
                "key": "+".join(present_keys),
                "n_rows_with_complete_key": int(valid_key_rows.sum()),
                "n_duplicated_key_rows": int(duplicates.sum()),
                "pct_duplicated_key_rows": float(duplicates.mean()) if len(duplicates) else 0.0,
            }
        ]
    )


def save_barplot(data: pd.DataFrame, x: str, y: str, output_path: Path, xlabel: str, ylabel: str, rotation: int = 0) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x=x, y=y, ax=ax)
    clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=rotation)
    save_figure(fig, output_path)


def plot_rows_by_year(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df.groupby("ano", dropna=False).size().reset_index(name="n")
    save_barplot(counts, "ano", "n", output_dir / "rows_by_year.png", "Year", "Number of records")


def plot_rows_by_state(df: pd.DataFrame, output_dir: Path) -> None:
    counts = df.groupby("sigla_uf", dropna=False).size().reset_index(name="n").sort_values("n", ascending=False)
    save_barplot(counts, "sigla_uf", "n", output_dir / "rows_by_state.png", "State", "Number of records", rotation=90)


def plot_age_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="idade", bins=70, ax=ax, color=sns.color_palette("colorblind")[0])
    clean_axis(ax)
    ax.set_xlabel("Age")
    ax.set_ylabel("Number of records")
    save_figure(fig, output_dir / "age_distribution.png")


def plot_year_state_heatmap(year_state_counts: pd.DataFrame, output_dir: Path) -> None:
    matrix = year_state_counts.pivot(index="sigla_uf", columns="ano", values="n").fillna(0)
    state_mean = matrix.replace(0, pd.NA).mean(axis=1)
    relative_matrix = matrix.div(state_mean, axis=0)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(
        relative_matrix,
        cmap="cividis",
        center=1.0,
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Deaths relative to state average"},
    )
    clean_axis(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("State")
    save_figure(fig, output_dir / "rows_by_year_and_state_heatmap.png")


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()

    dataset_summary = build_dataset_summary(df)
    column_summary = build_column_summary(df)
    year_state_counts = build_year_state_counts(df)
    date_checks = build_death_date_checks(df)
    duplicate_checks = build_duplicate_checks(df)

    dataset_summary.to_csv(output_dir / "dataset_summary.csv", index=False)
    column_summary.to_csv(output_dir / "column_sanity_summary.csv", index=False)
    year_state_counts.to_csv(output_dir / "rows_by_year_state.csv", index=False)
    date_checks.to_csv(output_dir / "death_date_checks.csv", index=False)
    duplicate_checks.to_csv(output_dir / "duplicate_key_checks.csv", index=False)

    plot_rows_by_year(df, output_dir)
    plot_rows_by_state(df, output_dir)
    plot_age_distribution(df, output_dir)
    plot_year_state_heatmap(year_state_counts, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sanity-check EDA for the processed SIM dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the processed dataset.csv file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where tables and figures will be saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_dataset(args.dataset)
    write_outputs(df, args.output_dir)
    print(f"Sanity EDA outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
