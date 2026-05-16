from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from plot_style import clean_axis, save_figure, set_plot_style
from sanity_check import DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_DIR, read_dataset

OUTCOME_COL = "morte_evitavel"
OUTCOME_LABELS = {
    0: "Non-avoidable",
    1: "Avoidable",
    2: "Ill-defined",
}

AGE_BINS = [5, 15, 25, 35, 45, 55, 65, 75]
AGE_LABELS = ["5-14", "15-24", "25-34", "35-44", "45-54", "55-64", "65-74"]

CATEGORY_LABELS = {
    "sexo": {
        "0": "Ignored",
        "1": "Male",
        "2": "Female",
    },
    "raca_cor": {
        "1": "White",
        "2": "Black",
        "3": "Asian",
        "4": "Brown",
        "5": "Indigenous",
    },
    "escolaridade_grupo": {
        "baixa": "Low",
        "media": "Medium",
        "alta": "High",
    },
}


def outcome_label(value: object) -> str:
    if pd.isna(value):
        return "Missing"

    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return str(value)

    return OUTCOME_LABELS.get(numeric_value, str(value))


def add_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["outcome_label"] = df[OUTCOME_COL].map(outcome_label)
    return df


def normalized_code_label(value: object, mapping: dict[str, str]) -> str:
    if pd.isna(value):
        return "Missing"

    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]

    return mapping.get(raw, raw)


def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_group"] = pd.cut(df["idade"], bins=AGE_BINS, labels=AGE_LABELS, right=False, include_lowest=True)
    return df


def count_and_share(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    counts = df.groupby(group_cols + ["outcome_label"], dropna=False, observed=False).size().reset_index(name="n")
    totals = counts.groupby(group_cols, dropna=False, observed=False)["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    return counts


def overall_distribution(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = df.groupby("outcome_label", dropna=False).size().reset_index(name="n")
    table["share"] = table["n"] / table["n"].sum()
    table.to_csv(output_dir / "outcome_distribution.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(data=table, x="outcome_label", y="share", ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Share of records")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    save_figure(fig, output_dir / "outcome_distribution.png")
    return table


def outcome_by_year(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = count_and_share(df, ["ano"])
    table.to_csv(output_dir / "outcome_by_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=table, x="ano", y="share", hue="outcome_label", marker="o", ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Year")
    ax.set_ylabel("Share of records")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / "outcome_by_year.png")
    return table


def outcome_by_state(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    table = count_and_share(df, ["sigla_uf"])
    table.to_csv(output_dir / "outcome_by_state.csv", index=False)

    avoidable = table[table["outcome_label"] == "Avoidable"].sort_values("share", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=avoidable, x="sigla_uf", y="share", ax=ax)
    clean_axis(ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Avoidable share")
    ax.tick_params(axis="x", rotation=90)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    save_figure(fig, output_dir / "outcome_avoidable_share_by_state.png")

    matrix = table.pivot(index="sigla_uf", columns="outcome_label", values="share").fillna(0)
    fig, ax = plt.subplots(figsize=(7, 8))
    sns.heatmap(matrix, cmap="cividis", linewidths=0.5, linecolor="white", ax=ax, cbar_kws={"label": "Share of state records"})
    clean_axis(ax)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("State")
    save_figure(fig, output_dir / "outcome_by_state_heatmap.png")
    return table


def outcome_by_age_group(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    df = add_age_group(df)
    table = count_and_share(df, ["age_group"])
    table.to_csv(output_dir / "outcome_by_age_group.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=table, x="age_group", y="share", hue="outcome_label", marker="o", ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Share of records")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / "outcome_by_age_group.png")
    return table


def outcome_by_category(df: pd.DataFrame, column: str, output_dir: Path) -> pd.DataFrame:
    data = df.copy()
    label_col = f"{column}_label"
    mapping = CATEGORY_LABELS.get(column, {})
    data[label_col] = data[column].map(lambda value: normalized_code_label(value, mapping))

    table = count_and_share(data, [label_col]).rename(columns={label_col: column})
    table.to_csv(output_dir / f"outcome_by_{column}.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=table, x=column, y="share", hue="outcome_label", ax=ax)
    clean_axis(ax)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Share of records")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / f"outcome_by_{column}.png")
    return table


def outcome_by_occupation(df: pd.DataFrame, output_dir: Path, top_n: int = 20) -> pd.DataFrame:
    data = df.copy()
    data["occupation_code"] = data["ocupacao"].map(lambda value: normalized_code_label(value, {}))
    occupation_counts = data["occupation_code"].value_counts(dropna=False).head(top_n).index
    data = data[data["occupation_code"].isin(occupation_counts)].copy()
    table = count_and_share(data, ["occupation_code"])
    totals = data.groupby("occupation_code", dropna=False).size().reset_index(name="total_n")
    table = table.merge(totals, on="occupation_code", how="left")
    table = table.sort_values(["total_n", "occupation_code", "outcome_label"], ascending=[False, True, True])
    table.to_csv(output_dir / "outcome_by_top_occupations.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(data=table, x="occupation_code", y="share", hue="outcome_label", ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Occupation code")
    ax.set_ylabel("Share of records")
    ax.tick_params(axis="x", rotation=90)
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / "outcome_by_top_occupations.png")
    return table


def cid_chapter(code: object) -> str:
    if pd.isna(code):
        return "Missing"

    text = str(code).strip().upper()
    if not text:
        return "Missing"

    return text[0]


def outcome_by_cid_chapter(df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    data = df.copy()
    data["cid_chapter_initial"] = data["causa_basica"].map(cid_chapter)
    table = count_and_share(data, ["cid_chapter_initial"])
    table.to_csv(output_dir / "outcome_by_cid_chapter_initial.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(data=table, x="cid_chapter_initial", y="share", hue="outcome_label", ax=ax)
    clean_axis(ax)
    ax.set_xlabel("ICD-10 chapter initial")
    ax.set_ylabel("Share of records")
    ax.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / "outcome_by_cid_chapter_initial.png")
    return table


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return slug or "value"


def top_causes_by_outcome(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> pd.DataFrame:
    rows = []
    for label, group in df.groupby("outcome_label", dropna=False):
        counts = group["causa_basica"].value_counts(dropna=False).head(top_n).reset_index()
        counts.columns = ["causa_basica", "n"]
        counts["outcome_label"] = label
        counts["share_within_outcome"] = counts["n"] / len(group)
        rows.append(counts)

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_data = counts.sort_values("n", ascending=True)
        sns.barplot(data=plot_data, x="share_within_outcome", y="causa_basica", ax=ax)
        clean_axis(ax)
        ax.set_xlabel("Share within outcome")
        ax.set_ylabel("Underlying cause")
        ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        save_figure(fig, output_dir / f"outcome_top_causes_{slugify(str(label))}.png")

    table = pd.concat(rows, ignore_index=True)
    table = table[["outcome_label", "causa_basica", "n", "share_within_outcome"]]
    table.to_csv(output_dir / "outcome_top_causes_by_outcome.csv", index=False)
    return table


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    if OUTCOME_COL not in df.columns:
        raise ValueError(f"Required column not found: {OUTCOME_COL}")

    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    df = add_outcome_label(df)

    overall_distribution(df, output_dir)
    outcome_by_year(df, output_dir)
    outcome_by_state(df, output_dir)
    outcome_by_age_group(df, output_dir)
    outcome_by_category(df, "sexo", output_dir)
    outcome_by_category(df, "raca_cor", output_dir)
    outcome_by_category(df, "escolaridade_grupo", output_dir)
    outcome_by_occupation(df, output_dir)
    outcome_by_cid_chapter(df, output_dir)
    top_causes_by_outcome(df, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the avoidable mortality outcome in the processed SIM dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the processed dataset.csv file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where tables and figures will be saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_dataset(args.dataset)
    write_outputs(df, args.output_dir)
    print(f"Outcome profile outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
