from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import expm
from scipy.optimize import minimize

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "sim_selected" / "dataset.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "discovery" / "notears"

DEFAULT_VARIABLES = [
    "ano",
    "sigla_uf",
    "idade",
    "sexo",
    "raca_cor",
    "ocupacao",
    "local_ocorrencia",
    "morte_evitavel",
    "escolaridade_grupo",
]

DATE_COLUMNS = {}
NUMERIC_COLUMNS = {"ano", "idade", "morte_evitavel"}
CATEGORICAL_COLUMNS = {
    "sigla_uf",
    "sexo",
    "raca_cor",
    "ocupacao",
    "estado_civil",
    "local_ocorrencia",
    "escolaridade_grupo",
}

CODE_COLUMNS = CATEGORICAL_COLUMNS | {"morte_evitavel"}


@dataclass(frozen=True)
class PreparedData:
    matrix: np.ndarray
    feature_names: list[str]
    feature_to_variable: dict[str, str]
    means: pd.Series
    stds: pd.Series


def normalize_code(value: object) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA

    text = str(value).strip()
    if text == "":
        return pd.NA

    try:
        numeric_value = float(text)
    except ValueError:
        return text

    if numeric_value.is_integer():
        return str(int(numeric_value))

    return text


def parse_variables(raw_variables: str | None) -> list[str]:
    if raw_variables is None:
        return DEFAULT_VARIABLES

    variables = [variable.strip() for variable in raw_variables.split(",")]
    return [variable for variable in variables if variable]


def read_dataset(path: Path, variables: Iterable[str]) -> pd.DataFrame:
    variables = list(variables)
    dtype = {column: "string" for column in CODE_COLUMNS if column in variables}
    return pd.read_csv(path, usecols=variables, dtype=dtype, low_memory=False)


def sample_rows(df: pd.DataFrame, sample_size: int | None, random_state: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.copy()

    return df.sample(n=sample_size, random_state=random_state).copy()


def prepare_date_column(series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(series, errors="coerce")
    return pd.Series(dates.map(lambda value: value.toordinal() if pd.notna(value) else pd.NA), index=series.index, dtype="Float64")


def cap_categories(series: pd.Series, max_categories: int) -> pd.Series:
    normalized = series.map(normalize_code).astype("string")
    normalized = normalized.fillna("__MISSING__")

    if max_categories <= 0:
        return normalized

    top_values = set(normalized.value_counts(dropna=False).head(max_categories).index.astype("string"))
    return normalized.where(normalized.isin(top_values), "__OTHER__")


def prepare_data(df: pd.DataFrame, max_categories: int) -> PreparedData:
    parts = []
    feature_to_variable: dict[str, str] = {}

    for column in df.columns:
        if column in DATE_COLUMNS:
            prepared = prepare_date_column(df[column]).rename(column).to_frame()
        elif column in NUMERIC_COLUMNS:
            prepared = pd.to_numeric(df[column], errors="coerce").rename(column).to_frame()
        else:
            capped = cap_categories(df[column], max_categories=max_categories)
            prepared = pd.get_dummies(capped, prefix=column, dtype=float)

        for feature in prepared.columns:
            feature_to_variable[feature] = column

        parts.append(prepared)

    encoded = pd.concat(parts, axis=1)
    encoded = encoded.apply(pd.to_numeric, errors="coerce")
    encoded = encoded.fillna(encoded.mean(numeric_only=True)).fillna(0.0)

    means = encoded.mean(axis=0)
    stds = encoded.std(axis=0, ddof=0).replace(0, 1.0)
    standardized = (encoded - means) / stds

    return PreparedData(
        matrix=standardized.to_numpy(dtype=float),
        feature_names=list(standardized.columns),
        feature_to_variable=feature_to_variable,
        means=means,
        stds=stds,
    )


def squared_loss(X: np.ndarray, W: np.ndarray) -> tuple[float, np.ndarray]:
    n = X.shape[0]
    residual = X @ W - X
    loss = 0.5 / n * np.sum(residual ** 2)
    gradient = X.T @ residual / n
    return loss, gradient


def acyclicity(W: np.ndarray) -> tuple[float, np.ndarray]:
    E = expm(W * W)
    h = np.trace(E) - W.shape[0]
    gradient = E.T * W * 2
    return h, gradient


def build_allowed_edge_mask(feature_names: list[str], feature_to_variable: dict[str, str], forbid_within_variable: bool) -> np.ndarray:
    d = len(feature_names)
    mask = np.ones((d, d), dtype=float)
    np.fill_diagonal(mask, 0.0)

    if forbid_within_variable:
        for i, source in enumerate(feature_names):
            source_variable = feature_to_variable[source]
            for j, target in enumerate(feature_names):
                if source_variable == feature_to_variable[target]:
                    mask[i, j] = 0.0

    return mask


def notears_linear(
    X: np.ndarray,
    lambda1: float,
    loss_type: str = "l2",
    max_iter: int = 100,
    h_tol: float = 1e-8,
    rho_max: float = 1e16,
    w_threshold: float = 0.3,
    allowed_edge_mask: np.ndarray | None = None,
) -> np.ndarray:
    if loss_type != "l2":
        raise ValueError("Only l2 loss is implemented.")

    _, d = X.shape
    if allowed_edge_mask is None:
        allowed_edge_mask = np.ones((d, d), dtype=float)
        np.fill_diagonal(allowed_edge_mask, 0.0)

    def unpack(w_split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        w_pos = w_split[: d * d].reshape(d, d)
        w_neg = w_split[d * d :].reshape(d, d)
        W_raw = w_pos - w_neg
        W = W_raw * allowed_edge_mask
        return W, w_pos, w_neg

    def objective(w_split: np.ndarray, rho: float, alpha: float) -> tuple[float, np.ndarray]:
        W, _, _ = unpack(w_split)
        loss, grad_loss = squared_loss(X, W)
        h, grad_h = acyclicity(W)
        smooth_objective = loss + 0.5 * rho * h * h + alpha * h
        smooth_gradient = grad_loss + (rho * h + alpha) * grad_h
        smooth_gradient = smooth_gradient * allowed_edge_mask

        objective_value = smooth_objective + lambda1 * np.sum(w_split)
        gradient_pos = smooth_gradient + lambda1
        gradient_neg = -smooth_gradient + lambda1
        gradient = np.concatenate([gradient_pos.ravel(), gradient_neg.ravel()])
        return objective_value, gradient

    bounds = [(0.0, None)] * (2 * d * d)
    w_est = np.zeros(2 * d * d)
    rho = 1.0
    alpha = 0.0
    h = np.inf

    for _ in range(max_iter):
        w_new = None
        h_new = None

        while rho < rho_max:
            result = minimize(
                fun=lambda w: objective(w, rho, alpha),
                x0=w_est,
                method="L-BFGS-B",
                jac=True,
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-12},
            )
            W_new, _, _ = unpack(result.x)
            h_new, _ = acyclicity(W_new)
            w_new = result.x

            if h_new <= 0.25 * h or h == np.inf:
                break

            rho *= 10

        if w_new is None or h_new is None:
            raise RuntimeError("NOTEARS optimization failed before producing an iterate.")

        w_est = w_new
        h = h_new
        alpha += rho * h

        if h <= h_tol or rho >= rho_max:
            break

    W_est, _, _ = unpack(w_est)
    W_est[np.abs(W_est) < w_threshold] = 0.0
    return W_est


def feature_edges_dataframe(W: np.ndarray, feature_names: list[str], feature_to_variable: dict[str, str]) -> pd.DataFrame:
    rows = []
    sources, targets = np.where(W != 0)

    for source, target in zip(sources, targets, strict=False):
        source_feature = feature_names[source]
        target_feature = feature_names[target]
        rows.append(
            {
                "source_feature": source_feature,
                "target_feature": target_feature,
                "source_variable": feature_to_variable[source_feature],
                "target_variable": feature_to_variable[target_feature],
                "weight": W[source, target],
                "abs_weight": abs(W[source, target]),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_weight", ascending=False) if rows else pd.DataFrame(columns=["source_feature", "target_feature", "source_variable", "target_variable", "weight", "abs_weight"])


def variable_edges_dataframe(feature_edges: pd.DataFrame) -> pd.DataFrame:
    if feature_edges.empty:
        return pd.DataFrame(columns=["source_variable", "target_variable", "n_feature_edges", "max_abs_weight", "mean_abs_weight", "signed_weight_at_max_abs"])

    rows = []
    for (source, target), group in feature_edges.groupby(["source_variable", "target_variable"]):
        max_index = group["abs_weight"].idxmax()
        rows.append(
            {
                "source_variable": source,
                "target_variable": target,
                "n_feature_edges": len(group),
                "max_abs_weight": group["abs_weight"].max(),
                "mean_abs_weight": group["abs_weight"].mean(),
                "signed_weight_at_max_abs": feature_edges.loc[max_index, "weight"],
            }
        )

    return pd.DataFrame(rows).sort_values("max_abs_weight", ascending=False)


def build_variable_graph(variable_edges: pd.DataFrame, weight_column: str = "max_abs_weight") -> nx.DiGraph:
    graph = nx.DiGraph()

    if variable_edges.empty:
        return graph

    variables = sorted(set(variable_edges["source_variable"]) | set(variable_edges["target_variable"]))
    graph.add_nodes_from(variables)

    for row in variable_edges.itertuples(index=False):
        edge_attrs = row._asdict()
        weight = float(edge_attrs.get(weight_column, 1.0))
        graph.add_edge(row.source_variable, row.target_variable, weight=weight, **edge_attrs)

    return graph


def save_graph_image(variable_edges: pd.DataFrame, output_path: Path, weight_column: str = "max_abs_weight") -> None:
    graph = build_variable_graph(variable_edges, weight_column=weight_column)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_axis_off()

    if graph.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "No edges after thresholding", ha="center", va="center", fontsize=12)
        fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return

    pos = nx.spring_layout(graph, seed=42, k=1.2)
    weights = [abs(graph[u][v].get("weight", 1.0)) for u, v in graph.edges()]
    max_weight = max(weights) if weights else 1.0
    widths = [1.0 + 4.0 * weight / max_weight for weight in weights]

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_color="#d9d9d9",
        edgecolors="#555555",
        linewidths=1.0,
        node_size=2600,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_color="#222222", ax=ax)
    nx.draw_networkx_edges(
        graph,
        pos,
        width=widths,
        edge_color="#4c78a8",
        arrows=True,
        arrowsize=18,
        min_source_margin=18,
        min_target_margin=18,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )

    edge_labels = {}
    for source, target in graph.edges():
        value = graph[source][target].get(weight_column)
        if value is not None:
            edge_labels[(source, target)] = f"{float(value):.2f}"

    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def save_graphml(variable_edges: pd.DataFrame, output_path: Path, weight_column: str = "max_abs_weight") -> None:
    graph = build_variable_graph(variable_edges, weight_column=weight_column)
    nx.write_graphml(graph, output_path)


def bootstrap_variable_edges(
    df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rng = np.random.default_rng(args.random_state)
    edge_counts: dict[tuple[str, str], int] = {}
    max_weights: dict[tuple[str, str], list[float]] = {}
    signed_weights: dict[tuple[str, str], list[float]] = {}

    bootstrap_size = args.bootstrap_sample_size or len(df)

    for bootstrap_idx in range(args.bootstrap_runs):
        sampled_indices = rng.choice(df.index.to_numpy(), size=bootstrap_size, replace=True)
        bootstrap_df = df.loc[sampled_indices].copy()
        prepared = prepare_data(bootstrap_df, max_categories=args.max_categories)
        allowed_edge_mask = build_allowed_edge_mask(
            prepared.feature_names,
            prepared.feature_to_variable,
            forbid_within_variable=args.forbid_within_variable,
        )
        W = notears_linear(
            prepared.matrix,
            lambda1=args.lambda1,
            max_iter=args.max_iter,
            h_tol=args.h_tol,
            w_threshold=args.w_threshold,
            allowed_edge_mask=allowed_edge_mask,
        )
        feature_edges = feature_edges_dataframe(W, prepared.feature_names, prepared.feature_to_variable)
        variable_edges = variable_edges_dataframe(feature_edges)

        seen_edges = set()
        for row in variable_edges.itertuples(index=False):
            edge = (row.source_variable, row.target_variable)
            seen_edges.add(edge)
            max_weights.setdefault(edge, []).append(float(row.max_abs_weight))
            signed_weights.setdefault(edge, []).append(float(row.signed_weight_at_max_abs))

        for edge in seen_edges:
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

        print(f"Bootstrap {bootstrap_idx + 1}/{args.bootstrap_runs}: {len(seen_edges)} variable-level edges")

    rows = []
    for (source, target), count in edge_counts.items():
        support = count / args.bootstrap_runs if args.bootstrap_runs else 0.0
        rows.append(
            {
                "source_variable": source,
                "target_variable": target,
                "bootstrap_count": count,
                "bootstrap_support": support,
                "mean_max_abs_weight": float(np.mean(max_weights[(source, target)])),
                "mean_signed_weight_at_max_abs": float(np.mean(signed_weights[(source, target)])),
            }
        )

    columns = [
        "source_variable",
        "target_variable",
        "bootstrap_count",
        "bootstrap_support",
        "mean_max_abs_weight",
        "mean_signed_weight_at_max_abs",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows).sort_values(["bootstrap_support", "mean_max_abs_weight"], ascending=False)


def adjacency_dataframe(W: np.ndarray, feature_names: list[str]) -> pd.DataFrame:
    return pd.DataFrame(W, index=feature_names, columns=feature_names)


def write_run_metadata(output_dir: Path, args: argparse.Namespace, prepared: PreparedData, n_rows: int) -> None:
    rows = [
        {"key": "dataset", "value": str(args.dataset)},
        {"key": "n_rows_used", "value": str(n_rows)},
        {"key": "n_features", "value": str(len(prepared.feature_names))},
        {"key": "variables", "value": ",".join(args.variables)},
        {"key": "sample_size", "value": str(args.sample_size)},
        {"key": "max_categories", "value": str(args.max_categories)},
        {"key": "lambda1", "value": str(args.lambda1)},
        {"key": "w_threshold", "value": str(args.w_threshold)},
        {"key": "max_iter", "value": str(args.max_iter)},
        {"key": "h_tol", "value": str(args.h_tol)},
        {"key": "forbid_within_variable", "value": str(args.forbid_within_variable)},
        {"key": "bootstrap_runs", "value": str(args.bootstrap_runs)},
        {"key": "bootstrap_sample_size", "value": str(args.bootstrap_sample_size)},
        {"key": "bootstrap_support_threshold", "value": str(args.bootstrap_support_threshold)},
    ]
    pd.DataFrame(rows).to_csv(output_dir / "run_metadata.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run linear NOTEARS graph discovery on the processed SIM dataset.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="Path to data/processed/sim_selected/dataset.csv.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for NOTEARS outputs.")
    parser.add_argument("--variables", type=str, default=None, help="Comma-separated variables to use. Defaults to the initial DAG discovery set.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Number of rows sampled before fitting. Use 0 or negative to use all rows.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for row sampling.")
    parser.add_argument("--max-categories", type=int, default=20, help="Maximum categories kept per categorical variable before __OTHER__ grouping.")
    parser.add_argument("--lambda1", type=float, default=0.05, help="L1 sparsity penalty.")
    parser.add_argument("--w-threshold", type=float, default=0.3, help="Absolute edge-weight threshold applied after optimization.")
    parser.add_argument("--max-iter", type=int, default=100, help="Maximum augmented-Lagrangian iterations.")
    parser.add_argument("--h-tol", type=float, default=1e-8, help="Acyclicity tolerance.")
    parser.add_argument("--forbid-within-variable", action=argparse.BooleanOptionalAction, default=True, help="Forbid edges between one-hot features generated from the same original variable.")
    parser.add_argument("--bootstrap-runs", type=int, default=50, help="Number of bootstrap NOTEARS fits used to estimate edge support.")
    parser.add_argument("--bootstrap-sample-size", type=int, default=None, help="Rows sampled with replacement per bootstrap. Defaults to the fitted sample size.")
    parser.add_argument("--bootstrap-support-threshold", type=float, default=0.5, help="Minimum bootstrap support retained in the consensus graph.")
    parser.add_argument("--skip-bootstrap", action="store_true", help="Skip bootstrap support estimation and only fit the main NOTEARS graph.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.variables = parse_variables(args.variables)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = read_dataset(args.dataset, args.variables)
    df = sample_rows(df, args.sample_size, args.random_state)
    prepared = prepare_data(df, max_categories=args.max_categories)

    allowed_edge_mask = build_allowed_edge_mask(
        prepared.feature_names,
        prepared.feature_to_variable,
        forbid_within_variable=args.forbid_within_variable,
    )

    W = notears_linear(
        prepared.matrix,
        lambda1=args.lambda1,
        max_iter=args.max_iter,
        h_tol=args.h_tol,
        w_threshold=args.w_threshold,
        allowed_edge_mask=allowed_edge_mask,
    )

    feature_edges = feature_edges_dataframe(W, prepared.feature_names, prepared.feature_to_variable)
    variable_edges = variable_edges_dataframe(feature_edges)
    adjacency = adjacency_dataframe(W, prepared.feature_names)

    feature_edges.to_csv(args.output_dir / "feature_edges.csv", index=False)
    variable_edges.to_csv(args.output_dir / "variable_edges.csv", index=False)
    adjacency.to_csv(args.output_dir / "feature_adjacency_matrix.csv")
    save_graph_image(variable_edges, args.output_dir / "variable_graph.png", weight_column="max_abs_weight")
    save_graphml(variable_edges, args.output_dir / "variable_graph.graphml", weight_column="max_abs_weight")
    pd.DataFrame(
        [{"feature": feature, "variable": prepared.feature_to_variable[feature]} for feature in prepared.feature_names]
    ).to_csv(args.output_dir / "feature_mapping.csv", index=False)
    pd.DataFrame(
        {
            "feature": prepared.feature_names,
            "mean": [prepared.means[feature] for feature in prepared.feature_names],
            "std": [prepared.stds[feature] for feature in prepared.feature_names],
        }
    ).to_csv(args.output_dir / "standardization_params.csv", index=False)

    bootstrap_edges = pd.DataFrame()
    consensus_edges = pd.DataFrame()
    if not args.skip_bootstrap and args.bootstrap_runs > 0:
        bootstrap_edges = bootstrap_variable_edges(df, args)
        consensus_edges = bootstrap_edges[bootstrap_edges["bootstrap_support"] >= args.bootstrap_support_threshold].copy()
        bootstrap_edges.to_csv(args.output_dir / "bootstrap_variable_edges.csv", index=False)
        consensus_edges.to_csv(args.output_dir / "bootstrap_consensus_variable_edges.csv", index=False)
        save_graph_image(consensus_edges, args.output_dir / "bootstrap_consensus_variable_graph.png", weight_column="bootstrap_support")
        save_graphml(consensus_edges, args.output_dir / "bootstrap_consensus_variable_graph.graphml", weight_column="bootstrap_support")

    write_run_metadata(args.output_dir, args, prepared, len(df))

    print(f"NOTEARS outputs saved to: {args.output_dir}")
    print(f"Rows used: {len(df):,}")
    print(f"Features used: {len(prepared.feature_names):,}")
    print(f"Feature-level edges: {len(feature_edges):,}")
    print(f"Variable-level edges: {len(variable_edges):,}")
    if not args.skip_bootstrap and args.bootstrap_runs > 0:
        print(f"Bootstrap runs: {args.bootstrap_runs:,}")
        print(f"Bootstrap edges: {len(bootstrap_edges):,}")
        print(f"Consensus edges with support >= {args.bootstrap_support_threshold:.0%}: {len(consensus_edges):,}")


if __name__ == "__main__":
    main()
