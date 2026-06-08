from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
from scipy.stats import chi2_contingency, combine_pvalues


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DAG_PATH = PROJECT_ROOT / "causal" / "dag_final.json"
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "sim_selected" / "dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "causal" / "output" / "final_dag_checks"

DEFAULT_AGE_BINS = [0, 25, 45, 60, 75]
DEFAULT_AGE_LABELS = ["jovem", "adulto", "meia-idade", "idoso"]


def load_dag_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    if "edges" not in config:
        raise ValueError(f"Arquivo de DAG sem chave obrigatoria 'edges': {path}")

    return config


def build_graph(edges: Iterable[Iterable[str]]) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_edges_from(tuple(edge) for edge in edges)
    return graph


def normalize_code_value(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if not text:
        return pd.NA

    try:
        numeric = float(text)
    except ValueError:
        return text

    if numeric.is_integer():
        return str(int(numeric))

    return text


def read_dataset(path: Path, graph_nodes: Iterable[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset nao encontrado: {path}. Gere data/processed/sim_selected/dataset.csv antes de rodar os testes."
        )

    df = pd.read_csv(path, low_memory=False)

    if "idade_grupo" in set(graph_nodes):
        if "idade" not in df.columns:
            raise ValueError("O DAG usa 'idade_grupo', mas o dataset nao tem a coluna original 'idade'.")

        idade = pd.to_numeric(df["idade"], errors="coerce")
        df["idade_grupo"] = pd.cut(
            idade,
            bins=DEFAULT_AGE_BINS,
            labels=DEFAULT_AGE_LABELS,
            right=False,
            include_lowest=True,
        )

    for column in set(graph_nodes):
        if column in df.columns:
            df[column] = df[column].map(normalize_code_value).astype("string")

    return df


def validate_dataset_columns(df: pd.DataFrame, graph: nx.DiGraph) -> None:
    missing = sorted(set(graph.nodes()) - set(df.columns))
    if missing:
        raise ValueError("Colunas do DAG ausentes no dataset: " + ", ".join(missing))


def validation_summary(graph: nx.DiGraph) -> pd.DataFrame:
    is_dag = nx.is_directed_acyclic_graph(graph)
    cycle = ""

    if not is_dag:
        try:
            cycle = repr(nx.find_cycle(graph))
        except nx.NetworkXNoCycle:
            cycle = ""

    topological_order = list(nx.topological_sort(graph)) if is_dag else []

    return pd.DataFrame(
        [
            {
                "is_dag": is_dag,
                "n_nodes": graph.number_of_nodes(),
                "n_edges": graph.number_of_edges(),
                "topological_order": " | ".join(topological_order),
                "cycle_if_any": cycle,
            }
        ]
    )


def dag_implications(graph: nx.DiGraph) -> pd.DataFrame:
    rows = []

    def adjacent(a: str, b: str) -> bool:
        return graph.has_edge(a, b) or graph.has_edge(b, a)

    for x, y in combinations(sorted(graph.nodes()), 2):
        if adjacent(x, y):
            continue

        separator = nx.find_minimal_d_separator(graph, x, y)
        if separator is None:
            continue

        rows.append(
            {
                "x": x,
                "y": y,
                "conditioning_set": "|".join(sorted(separator)),
                "conditioning_set_size": len(separator),
            }
        )

    return pd.DataFrame(rows)


def contingency_pvalue(data: pd.DataFrame, x: str, y: str) -> tuple[float, int, int]:
    table = pd.crosstab(data[x], data[y], dropna=False)

    if table.shape[0] < 2 or table.shape[1] < 2:
        return 1.0, int(table.to_numpy().sum()), int(table.size)

    try:
        _, p_value, _, _ = chi2_contingency(table, correction=False)
    except ValueError:
        p_value = 1.0

    return float(p_value), int(table.to_numpy().sum()), int(table.size)


def conditional_independence_pvalue(
    data: pd.DataFrame,
    x: str,
    y: str,
    conditioning_set: list[str],
    min_stratum_size: int,
) -> dict:
    required = [x, y] + conditioning_set
    clean = data[required].dropna().copy()

    if clean.empty:
        return {
            "p_value": 1.0,
            "n_used": 0,
            "n_strata_tested": 0,
            "n_strata_skipped": 0,
            "method": "empty",
        }

    if not conditioning_set:
        p_value, n_used, _ = contingency_pvalue(clean, x, y)
        return {
            "p_value": p_value,
            "n_used": n_used,
            "n_strata_tested": 1,
            "n_strata_skipped": 0,
            "method": "chi_square",
        }

    p_values = []
    n_used = 0
    n_skipped = 0

    groupby_key = conditioning_set[0] if len(conditioning_set) == 1 else conditioning_set
    for _, stratum in clean.groupby(groupby_key, dropna=False, observed=False):
        if len(stratum) < min_stratum_size:
            n_skipped += 1
            continue

        p_value, stratum_n, _ = contingency_pvalue(stratum, x, y)
        p_values.append(p_value)
        n_used += stratum_n

    if not p_values:
        return {
            "p_value": 1.0,
            "n_used": 0,
            "n_strata_tested": 0,
            "n_strata_skipped": n_skipped,
            "method": "no_testable_strata",
        }

    _, combined_p = combine_pvalues(p_values, method="fisher")

    return {
        "p_value": float(combined_p),
        "n_used": int(n_used),
        "n_strata_tested": len(p_values),
        "n_strata_skipped": n_skipped,
        "method": "fisher_combined_conditional_chi_square",
    }


def run_independence_tests(
    df: pd.DataFrame,
    implications: pd.DataFrame,
    alpha: float,
    n_bootstraps: int,
    sample_size: int | None,
    min_stratum_size: int,
    random_seed: int,
) -> pd.DataFrame:
    if implications.empty:
        return pd.DataFrame()

    alpha_bonferroni = alpha / len(implications)
    rows = []

    for implication_index, implication in implications.reset_index(drop=True).iterrows():
        x = implication["x"]
        y = implication["y"]
        conditioning_set = [
            item for item in str(implication["conditioning_set"]).split("|") if item
        ]

        p_values = []
        pass_count = 0
        n_used_values = []
        n_strata_tested_values = []
        n_strata_skipped_values = []
        methods = set()

        for boot_index in range(n_bootstraps):
            if sample_size is not None and sample_size < len(df):
                sample = df.sample(
                    n=sample_size,
                    replace=False,
                    random_state=random_seed + implication_index * n_bootstraps + boot_index,
                )
            else:
                sample = df

            result = conditional_independence_pvalue(
                sample,
                x=x,
                y=y,
                conditioning_set=conditioning_set,
                min_stratum_size=min_stratum_size,
            )
            p_value = result["p_value"]
            p_values.append(p_value)
            pass_count += int(p_value > alpha_bonferroni)
            n_used_values.append(result["n_used"])
            n_strata_tested_values.append(result["n_strata_tested"])
            n_strata_skipped_values.append(result["n_strata_skipped"])
            methods.add(result["method"])

        rows.append(
            {
                "x": x,
                "y": y,
                "conditioning_set": "|".join(conditioning_set),
                "alpha": alpha,
                "alpha_bonferroni": alpha_bonferroni,
                "n_bootstraps": n_bootstraps,
                "sample_size": sample_size if sample_size is not None else len(df),
                "min_stratum_size": min_stratum_size,
                "pass_rate": pass_count / n_bootstraps,
                "refuted": pass_count / n_bootstraps < 0.5,
                "median_p_value": float(pd.Series(p_values).median()),
                "min_p_value": float(pd.Series(p_values).min()),
                "max_p_value": float(pd.Series(p_values).max()),
                "mean_n_used": float(pd.Series(n_used_values).mean()),
                "mean_n_strata_tested": float(pd.Series(n_strata_tested_values).mean()),
                "mean_n_strata_skipped": float(pd.Series(n_strata_skipped_values).mean()),
                "method": "|".join(sorted(methods)),
            }
        )

    return pd.DataFrame(rows).sort_values(["refuted", "pass_rate"], ascending=[False, True])


def adjustment_set_backdoor(graph: nx.DiGraph, treatment: str, outcome: str) -> set[str] | None:
    if treatment not in graph or outcome not in graph:
        return None

    if outcome not in nx.descendants(graph, treatment):
        return None

    allowed = set(graph.nodes()) - (nx.descendants(graph, treatment) | {treatment})
    backdoor_graph = graph.copy()
    backdoor_graph.remove_edges_from(list(graph.out_edges(treatment)))

    try:
        separator = nx.find_minimal_d_separator(
            backdoor_graph,
            treatment,
            outcome,
            restricted=allowed,
        )
    except nx.NetworkXError:
        return None

    return set(separator) if separator is not None else None


def adjustment_sets_summary(graph: nx.DiGraph, treatments: list[str], outcome: str) -> pd.DataFrame:
    rows = []

    for treatment in treatments:
        adjustment_set = adjustment_set_backdoor(graph, treatment, outcome)
        rows.append(
            {
                "treatment": treatment,
                "outcome": outcome,
                "is_cause_of_outcome": outcome in nx.descendants(graph, treatment),
                "adjustment_set": "" if adjustment_set is None else "|".join(sorted(adjustment_set)),
                "adjustment_set_size": pd.NA if adjustment_set is None else len(adjustment_set),
            }
        )

    return pd.DataFrame(rows)


def positivity_diagnostics(
    df: pd.DataFrame,
    treatment: str,
    adjustment_set: list[str],
    min_cell_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = [treatment] + adjustment_set
    clean = df[required].dropna().copy()

    if clean.empty:
        summary = pd.DataFrame(
            [
                {
                    "treatment": treatment,
                    "adjustment_set": "|".join(adjustment_set),
                    "n_rows": 0,
                    "n_strata": 0,
                    "n_treatment_levels": 0,
                    "min_cell_count": min_cell_count,
                    "n_strata_missing_treatment_level": 0,
                    "pct_strata_missing_treatment_level": 0.0,
                    "n_strata_with_small_cell": 0,
                    "pct_strata_with_small_cell": 0.0,
                    "min_stratum_n": 0,
                    "median_stratum_n": 0,
                }
            ]
        )
        return summary, pd.DataFrame()

    treatment_levels = sorted(clean[treatment].dropna().unique().tolist())

    if adjustment_set:
        counts = (
            clean.groupby(adjustment_set + [treatment], dropna=False, observed=False)
            .size()
            .reset_index(name="n")
        )
        index_cols = adjustment_set
    else:
        counts = clean.groupby(treatment, dropna=False, observed=False).size().reset_index(name="n")
        counts["__overall__"] = "overall"
        index_cols = ["__overall__"]

    stratum_summary = (
        counts.groupby(index_cols, dropna=False, observed=False)
        .agg(
            stratum_n=("n", "sum"),
            n_present_treatment_levels=(treatment, "nunique"),
            min_observed_treatment_cell_n=("n", "min"),
        )
        .reset_index()
    )
    stratum_summary["n_global_treatment_levels"] = len(treatment_levels)
    stratum_summary["has_all_treatment_levels"] = (
        stratum_summary["n_present_treatment_levels"] == len(treatment_levels)
    )
    stratum_summary["has_small_observed_cell"] = (
        stratum_summary["min_observed_treatment_cell_n"] < min_cell_count
    )

    n_strata = len(stratum_summary)
    summary = pd.DataFrame(
        [
            {
                "treatment": treatment,
                "adjustment_set": "|".join(adjustment_set),
                "n_rows": len(clean),
                "n_strata": n_strata,
                "n_treatment_levels": len(treatment_levels),
                "treatment_levels": "|".join(map(str, treatment_levels)),
                "min_cell_count": min_cell_count,
                "n_strata_missing_treatment_level": int((~stratum_summary["has_all_treatment_levels"]).sum()),
                "pct_strata_missing_treatment_level": float((~stratum_summary["has_all_treatment_levels"]).mean()) if n_strata else 0.0,
                "n_strata_with_small_observed_cell": int(stratum_summary["has_small_observed_cell"].sum()),
                "pct_strata_with_small_observed_cell": float(stratum_summary["has_small_observed_cell"].mean()) if n_strata else 0.0,
                "min_stratum_n": int(stratum_summary["stratum_n"].min()) if n_strata else 0,
                "median_stratum_n": float(stratum_summary["stratum_n"].median()) if n_strata else 0.0,
            }
        ]
    )

    return summary, counts


def run_positivity_checks(
    df: pd.DataFrame,
    checks: list[dict],
    min_cell_count: int,
    output_dir: Path,
) -> pd.DataFrame:
    summaries = []

    for check in checks:
        treatment = check["treatment"]
        adjustment_set = check.get("adjustment_set", [])
        label = check.get("label", treatment)

        summary, cells = positivity_diagnostics(
            df,
            treatment=treatment,
            adjustment_set=adjustment_set,
            min_cell_count=min_cell_count,
        )
        summary.insert(0, "label", label)
        summaries.append(summary)

        if not cells.empty:
            cells.to_csv(output_dir / f"positivity_cells_{label}.csv", index=False)

    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Valida e testa empiricamente as implicacoes do DAG final."
    )
    parser.add_argument("--dag", type=Path, default=DEFAULT_DAG_PATH, help="Arquivo JSON com o DAG.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="CSV tratado do SIM.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretorio de saida.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia antes de Bonferroni.")
    parser.add_argument("--n-bootstraps", type=int, default=200, help="Numero de subamostras por implicacao.")
    parser.add_argument("--sample-size", type=int, default=5000, help="Tamanho da subamostra; use 0 para base completa.")
    parser.add_argument("--min-stratum-size", type=int, default=30, help="Tamanho minimo de estrato para teste condicional.")
    parser.add_argument("--min-cell-count", type=int, default=20, help="Contagem minima por celula para positividade.")
    parser.add_argument("--random-seed", type=int, default=42, help="Semente para subamostragem.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = load_dag_config(args.dag)
    graph = build_graph(config["edges"])

    validation = validation_summary(graph)
    validation.to_csv(args.output_dir / "dag_validation.csv", index=False)

    if not bool(validation.loc[0, "is_dag"]):
        raise ValueError("O grafo informado nao e aciclico. Veja dag_validation.csv.")

    implications = dag_implications(graph)
    implications.to_csv(args.output_dir / "independence_implications.csv", index=False)

    adjustment_sets = adjustment_sets_summary(
        graph,
        treatments=config.get("treatments", []),
        outcome=config.get("outcome", "morte_evitavel"),
    )
    adjustment_sets.to_csv(args.output_dir / "adjustment_sets.csv", index=False)

    df = read_dataset(args.dataset, graph.nodes())
    validate_dataset_columns(df, graph)
    df_model = df[list(graph.nodes())].dropna().copy()
    pd.DataFrame(
        [
            {
                "dataset_path": str(args.dataset),
                "n_rows_original": len(df),
                "n_rows_complete_for_dag": len(df_model),
                "n_columns_in_dag": len(graph.nodes()),
                "dag_columns": "|".join(graph.nodes()),
            }
        ]
    ).to_csv(args.output_dir / "analysis_rows_summary.csv", index=False)

    sample_size = None if args.sample_size == 0 else args.sample_size
    independence_tests = run_independence_tests(
        df_model,
        implications=implications,
        alpha=args.alpha,
        n_bootstraps=args.n_bootstraps,
        sample_size=sample_size,
        min_stratum_size=args.min_stratum_size,
        random_seed=args.random_seed,
    )
    independence_tests.to_csv(args.output_dir / "independence_tests.csv", index=False)

    positivity_summary = run_positivity_checks(
        df_model,
        checks=config.get("positivity_checks", []),
        min_cell_count=args.min_cell_count,
        output_dir=args.output_dir,
    )
    positivity_summary.to_csv(args.output_dir / "positivity_summary.csv", index=False)

    print("Checks do DAG final concluidos.")
    print(f"Saidas salvas em: {args.output_dir}")
    print(f"Implicacoes testaveis: {len(implications)}")
    if not independence_tests.empty:
        print(f"Implicacoes refutadas: {int(independence_tests['refuted'].sum())}")


if __name__ == "__main__":
    main()
