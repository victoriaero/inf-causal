from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from outcome_profile import add_age_group, add_outcome_label, count_and_share
from plot_style import clean_axis, save_figure, set_plot_style
from sanity_check import DEFAULT_DATASET_PATH, DEFAULT_OUTPUT_DIR, read_dataset

VALID_SEX_ORDER = ["Male", "Female"]
RACE_ORDER = ["White", "Black", "Asian", "Brown", "Indigenous"]
EDUCATION_ORDER = ["Low", "Medium", "High"]
MARITAL_ORDER = ["Single", "Married", "Widowed", "Separated", "Consensual union"]
OUTCOME_ORDER = ["Avoidable", "Ill-defined", "Non-avoidable"]

CATEGORY_LABELS = {
    "sexo": {
        "1": "Male",
        "2": "Female",
        "0": "Invalid/unknown",
    },
    "raca_cor": {
        "1": "White",
        "2": "Black",
        "3": "Asian",
        "4": "Brown",
        "5": "Indigenous",
        "0": "Invalid/unknown",
    },
    "escolaridade_grupo": {
        "baixa": "Low",
        "media": "Medium",
        "alta": "High",
    },
    "estado_civil": {
        "1": "Single",
        "2": "Married",
        "3": "Widowed",
        "4": "Separated",
        "5": "Consensual union",
        "0": "Invalid/unknown",
    },
}


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return "Missing"

    raw = str(value).strip()
    if raw.endswith(".0"):
        raw = raw[:-2]

    return raw or "Missing"


def label_column(df: pd.DataFrame, column: str, label_col: str | None = None) -> pd.DataFrame:
    df = df.copy()
    label_col = label_col or f"{column}_label"
    mapping = CATEGORY_LABELS.get(column, {})
    df[label_col] = df[column].map(lambda value: mapping.get(normalize_code(value), normalize_code(value)))
    return df


def ordered_subset(data: pd.DataFrame, column: str, order: list[str]) -> pd.DataFrame:
    return data[data[column].isin(order)].copy()


def save_grouped_barplot(data: pd.DataFrame, x: str, output_path: Path, xlabel: str, order: list[str] | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(data=data, x=x, y="share", hue="outcome_label", hue_order=OUTCOME_ORDER, order=order, ax=ax)
    clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of records")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_path)


def save_avoidable_heatmap(table: pd.DataFrame, index: str, columns: str, output_path: Path, xlabel: str, ylabel: str) -> None:
    avoidable = table[table["outcome_label"] == "Avoidable"]
    matrix = avoidable.pivot(index=index, columns=columns, values="share")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(
        matrix,
        cmap="cividis",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Avoidable share"},
    )
    clean_axis(ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    save_figure(fig, output_path)


def age_by_outcome(df: pd.DataFrame, output_dir: Path) -> None:
    data = add_age_group(df)
    table = count_and_share(data, ["age_group"])
    table.to_csv(output_dir / "social_age_group_by_outcome.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=table, x="age_group", y="share", hue="outcome_label", hue_order=OUTCOME_ORDER, marker="o", ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Age group")
    ax.set_ylabel("Share of records")
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / "social_age_group_by_outcome.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.boxplot(data=df, x="outcome_label", y="idade", order=OUTCOME_ORDER, showfliers=False, ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Age")
    save_figure(fig, output_dir / "social_age_by_outcome_boxplot.png")


def category_by_outcome(df: pd.DataFrame, column: str, output_dir: Path, order: list[str], xlabel: str) -> pd.DataFrame:
    label_col = f"{column}_label"
    data = label_column(df, column, label_col)
    table = count_and_share(data, [label_col]).rename(columns={label_col: column})
    table.to_csv(output_dir / f"social_{column}_by_outcome.csv", index=False)

    plot_data = ordered_subset(table, column, order)
    save_grouped_barplot(plot_data, column, output_dir / f"social_{column}_by_outcome.png", xlabel, order=order)
    return table


def occupation_group(value: object) -> str:
    code = normalize_code(value)
    if code in {"Missing", ""}:
        return "Missing"

    if code.startswith("999") or code.startswith("998"):
        return "Unclassified/unknown"

    digits = "".join(character for character in code if character.isdigit())
    if len(digits) < 2:
        return "Other"

    return f"{digits[:2]}"


def occupation_by_outcome(df: pd.DataFrame, output_dir: Path, top_n: int = 15) -> None:
    data = df.copy()
    data["occupation_group"] = data["ocupacao"].map(occupation_group)
    totals = data.groupby("occupation_group", dropna=False).size().reset_index(name="total_n")
    top_groups = totals.sort_values("total_n", ascending=False).head(top_n)["occupation_group"]
    data = data[data["occupation_group"].isin(top_groups)].copy()

    table = count_and_share(data, ["occupation_group"])
    table = table.merge(totals, on="occupation_group", how="left")
    table = table.sort_values(["total_n", "occupation_group", "outcome_label"], ascending=[False, True, True])
    table.to_csv(output_dir / "social_occupation_group_by_outcome.csv", index=False)

    order = table[["occupation_group", "total_n"]].drop_duplicates().sort_values("total_n", ascending=False)["occupation_group"].tolist()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=table, x="occupation_group", y="share", hue="outcome_label", hue_order=OUTCOME_ORDER, order=order, ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Occupation group")
    ax.set_ylabel("Share of records")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / "social_occupation_group_by_outcome.png")

    valid_data = data[~data["occupation_group"].isin(["Unclassified/unknown", "Missing", "Other"])].copy()
    valid_totals = valid_data.groupby("occupation_group", dropna=False).size().reset_index(name="total_n")
    valid_top_groups = valid_totals.sort_values("total_n", ascending=False).head(top_n)["occupation_group"]
    valid_data = valid_data[valid_data["occupation_group"].isin(valid_top_groups)].copy()
    valid_table = count_and_share(valid_data, ["occupation_group"])
    valid_table = valid_table.merge(valid_totals, on="occupation_group", how="left")
    valid_table = valid_table.sort_values(["total_n", "occupation_group", "outcome_label"], ascending=[False, True, True])
    valid_table.to_csv(output_dir / "social_occupation_group_valid_by_outcome.csv", index=False)

    valid_order = valid_table[["occupation_group", "total_n"]].drop_duplicates().sort_values("total_n", ascending=False)["occupation_group"].tolist()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=valid_table, x="occupation_group", y="share", hue="outcome_label", hue_order=OUTCOME_ORDER, order=valid_order, ax=ax)
    clean_axis(ax)
    ax.set_xlabel("Occupation group")
    ax.set_ylabel("Share of records")
    ax.tick_params(axis="x", rotation=45)
    ax.yaxis.set_major_formatter(lambda value, _: f"{value:.0%}")
    ax.legend(title="Outcome")
    save_figure(fig, output_dir / "social_occupation_group_valid_by_outcome.png")


def education_race_interaction(df: pd.DataFrame, output_dir: Path) -> None:
    data = label_column(label_column(df, "escolaridade_grupo", "education"), "raca_cor", "race")
    data = data[data["education"].isin(EDUCATION_ORDER) & data["race"].isin(RACE_ORDER)].copy()
    table = count_and_share(data, ["education", "race"])
    table.to_csv(output_dir / "social_interaction_education_race_outcome.csv", index=False)
    save_avoidable_heatmap(table, "race", "education", output_dir / "social_interaction_education_race_avoidable_heatmap.png", "Education", "Race/color")


def education_sex_interaction(df: pd.DataFrame, output_dir: Path) -> None:
    data = label_column(label_column(df, "escolaridade_grupo", "education"), "sexo", "sex")
    data = data[data["education"].isin(EDUCATION_ORDER) & data["sex"].isin(VALID_SEX_ORDER)].copy()
    table = count_and_share(data, ["education", "sex"])
    table.to_csv(output_dir / "social_interaction_education_sex_outcome.csv", index=False)
    save_avoidable_heatmap(table, "sex", "education", output_dir / "social_interaction_education_sex_avoidable_heatmap.png", "Education", "Sex")


def race_state_interaction(df: pd.DataFrame, output_dir: Path) -> None:
    data = label_column(df, "raca_cor", "race")
    data = data[data["race"].isin(RACE_ORDER)].copy()
    table = count_and_share(data, ["race", "sigla_uf"])
    table.to_csv(output_dir / "social_interaction_race_state_outcome.csv", index=False)

    avoidable = table[table["outcome_label"] == "Avoidable"]
    matrix = avoidable.pivot(index="race", columns="sigla_uf", values="share")

    fig, ax = plt.subplots(figsize=(13, 4.8))
    sns.heatmap(
        matrix,
        cmap="cividis",
        linewidths=0.5,
        linecolor="white",
        ax=ax,
        cbar_kws={"label": "Avoidable share"},
    )
    clean_axis(ax)
    ax.set_xlabel("State")
    ax.set_ylabel("Race/color")
    save_figure(fig, output_dir / "social_interaction_race_state_avoidable_heatmap.png")


def invalid_category_counts(df: pd.DataFrame, output_dir: Path) -> None:
    rows = []
    for column in ["sexo", "raca_cor", "estado_civil"]:
        labeled = label_column(df, column, "label")
        invalid = labeled[labeled["label"].isin(["Invalid/unknown", "Missing"])]
        rows.append({"column": column, "n": len(invalid), "share": len(invalid) / len(df) if len(df) else 0.0})

    pd.DataFrame(rows).to_csv(output_dir / "social_invalid_category_counts.csv", index=False)


def write_outputs(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    df = add_outcome_label(df)

    invalid_category_counts(df, output_dir)
    age_by_outcome(df, output_dir)
    category_by_outcome(df, "sexo", output_dir, VALID_SEX_ORDER, "Sex")
    category_by_outcome(df, "raca_cor", output_dir, RACE_ORDER, "Race/color")
    category_by_outcome(df, "escolaridade_grupo", output_dir, EDUCATION_ORDER, "Education")
    category_by_outcome(df, "estado_civil", output_dir, MARITAL_ORDER, "Marital status")
    occupation_by_outcome(df, output_dir)
    education_race_interaction(df, output_dir)
    education_sex_interaction(df, output_dir)
    race_state_interaction(df, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile social and demographic variables by avoidable mortality outcome.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to the processed dataset.csv file.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory where tables and figures will be saved.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = read_dataset(args.dataset)
    write_outputs(df, args.output_dir)
    print(f"Social demographic profile outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
