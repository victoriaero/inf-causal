from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from notears import save_graph_image, save_graphml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "processed" / "discovery" / "notears_less_sparse"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "filtered"

ROOT_VARIABLES = {
    "ano",
    "sigla_uf",
    "sexo",
    "raca_cor",
    "idade",
}

TERMINAL_VARIABLES = {
    "morte_evitavel",
}

TEMPORAL_TIERS = {
    "ano": 0,
    "sigla_uf": 0,
    "sexo": 0,
    "raca_cor": 0,
    "idade": 0,
    "escolaridade": 1,
    "escolaridade_grupo": 1,
    "ocupacao": 2,
    "estado_civil": 2,
    "local_ocorrencia": 3,
    "morte_evitavel": 4,
}

DERIVED_OR_REDUNDANT_PAIRS = {
    frozenset(("escolaridade", "escolaridade_grupo")),
    frozenset(("data_nascimento", "idade")),
    frozenset(("data_obito", "ano")),
}

MANUAL_FORBIDDEN_EDGES = {
    ("local_ocorrencia", "idade"),
    ("local_ocorrencia", "sexo"),
    ("local_ocorrencia", "sigla_uf"),
    ("local_ocorrencia", "raca_cor"),
    ("ocupacao", "idade"),
    ("ocupacao", "sexo"),
    ("ocupacao", "raca_cor"),
    ("ocupacao", "sigla_uf"),
    ("escolaridade_grupo", "idade"),
    ("escolaridade", "idade"),
    ("raca_cor", "idade"),
    ("idade", "ano"),
}

PREFERRED_BIDIRECTIONAL_DIRECTIONS = {
    frozenset(("sexo", "ocupacao")): ("sexo", "ocupacao"),
    frozenset(("idade", "ocupacao")): ("idade", "ocupacao"),
    frozenset(("idade", "local_ocorrencia")): ("idade", "local_ocorrencia"),
    frozenset(("sigla_uf", "raca_cor")): ("sigla_uf", "raca_cor"),
    frozenset(("local_ocorrencia", "morte_evitavel")): ("local_ocorrencia", "morte_evitavel"),
    frozenset(("escolaridade_grupo", "ocupacao")): ("escolaridade_grupo", "ocupacao"),
    frozenset(("ano", "escolaridade_grupo")): ("ano", "escolaridade_grupo"),
    frozenset(("sigla_uf", "ocupacao")): ("sigla_uf", "ocupacao"),
    frozenset(("sigla_uf", "escolaridade_grupo")): ("sigla_uf", "escolaridade_grupo"),
}


def read_edges(input_dir: Path, prefer_bootstrap: bool) -> tuple[pd.DataFrame, str, str]:
    bootstrap_path = input_dir / "bootstrap_consensus_variable_edges.csv"
    variable_path = input_dir / "variable_edges.csv"

    if prefer_bootstrap and bootstrap_path.exists():
        edges = pd.read_csv(bootstrap_path)
        return edges, "bootstrap", "bootstrap_support"

    if not variable_path.exists():
        raise FileNotFoundError(f"Could not find variable_edges.csv in {input_dir}")

    edges = pd.read_csv(variable_path)
    return edges, "main", "max_abs_weight"


def edge_strength(row: pd.Series, weight_column: str) -> float:
    value = row.get(weight_column)
    if pd.isna(value):
        return 0.0
    return float(value)


def rejection_reason(source: str, target: str) -> str | None:
    if source == target:
        return "self_edge"

    if frozenset((source, target)) in DERIVED_OR_REDUNDANT_PAIRS:
        return "derived_or_redundant_pair"

    if (source, target) in MANUAL_FORBIDDEN_EDGES:
        return "manual_forbidden_direction"

    if target in ROOT_VARIABLES:
        return "root_variable_cannot_have_incoming_edges"

    if source in TERMINAL_VARIABLES:
        return "terminal_variable_cannot_have_outgoing_edges"

    source_tier = TEMPORAL_TIERS.get(source)
    target_tier = TEMPORAL_TIERS.get(target)
    if source_tier is not None and target_tier is not None and source_tier > target_tier:
        return "violates_temporal_tier_order"

    return None


def apply_plausibility_rules(edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_rows = []
    rejected_rows = []

    for _, row in edges.iterrows():
        source = row["source_variable"]
        target = row["target_variable"]
        reason = rejection_reason(source, target)

        if reason is None:
            kept_rows.append(row.to_dict() | {"filter_status": "kept", "filter_reason": "plausible"})
        else:
            rejected_rows.append(row.to_dict() | {"filter_status": "rejected", "filter_reason": reason})

    kept = pd.DataFrame(kept_rows)
    rejected = pd.DataFrame(rejected_rows)
    return kept, rejected


def resolve_bidirectional_edges(edges: pd.DataFrame, weight_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if edges.empty:
        return edges.copy(), pd.DataFrame()

    edge_lookup = {(row.source_variable, row.target_variable): row for row in edges.itertuples(index=False)}
    processed_pairs = set()
    final_rows = []
    ambiguous_rows = []

    for row in edges.itertuples(index=False):
        source = row.source_variable
        target = row.target_variable
        pair = frozenset((source, target))

        if pair in processed_pairs:
            continue
        processed_pairs.add(pair)

        reverse_key = (target, source)
        current_key = (source, target)

        if reverse_key not in edge_lookup:
            final_rows.append(row._asdict() | {"bidirectional_resolution": "not_bidirectional"})
            continue

        current = edge_lookup[current_key]._asdict()
        reverse = edge_lookup[reverse_key]._asdict()
        preferred = PREFERRED_BIDIRECTIONAL_DIRECTIONS.get(pair)

        if preferred is not None:
            chosen = current if current_key == preferred else reverse
            dropped = reverse if current_key == preferred else current
            final_rows.append(chosen | {"bidirectional_resolution": "kept_preferred_direction"})
            ambiguous_rows.append(dropped | {"bidirectional_resolution": "dropped_opposite_of_preferred_direction"})
            continue

        current_strength = edge_strength(pd.Series(current), weight_column)
        reverse_strength = edge_strength(pd.Series(reverse), weight_column)

        if current_strength > reverse_strength:
            final_rows.append(current | {"bidirectional_resolution": "kept_stronger_direction"})
            ambiguous_rows.append(reverse | {"bidirectional_resolution": "dropped_weaker_reverse_direction"})
        elif reverse_strength > current_strength:
            final_rows.append(reverse | {"bidirectional_resolution": "kept_stronger_direction"})
            ambiguous_rows.append(current | {"bidirectional_resolution": "dropped_weaker_reverse_direction"})
        else:
            ambiguous_rows.append(current | {"bidirectional_resolution": "ambiguous_equal_strength"})
            ambiguous_rows.append(reverse | {"bidirectional_resolution": "ambiguous_equal_strength"})

    final = pd.DataFrame(final_rows)
    ambiguous = pd.DataFrame(ambiguous_rows)
    return final, ambiguous


def apply_thresholds(edges: pd.DataFrame, weight_column: str, min_weight: float | None, min_support: float | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    kept_rows = []
    rejected_rows = []

    for _, row in edges.iterrows():
        if min_support is not None and "bootstrap_support" in row and pd.notna(row["bootstrap_support"]):
            if float(row["bootstrap_support"]) < min_support:
                rejected_rows.append(row.to_dict() | {"filter_status": "rejected", "filter_reason": "below_min_bootstrap_support"})
                continue

        if min_weight is not None and weight_column in row and pd.notna(row[weight_column]):
            if abs(float(row[weight_column])) < min_weight:
                rejected_rows.append(row.to_dict() | {"filter_status": "rejected", "filter_reason": "below_min_weight"})
                continue

        kept_rows.append(row.to_dict())

    return pd.DataFrame(kept_rows), pd.DataFrame(rejected_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter NOTEARS variable-level graph using causal plausibility and bidirectional-edge rules.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Directory containing NOTEARS variable_edges.csv or bootstrap_consensus_variable_edges.csv.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for filtered graph outputs. Defaults to INPUT_DIR/filtered.")
    parser.add_argument("--prefer-bootstrap", action=argparse.BooleanOptionalAction, default=True, help="Use bootstrap_consensus_variable_edges.csv when available.")
    parser.add_argument("--min-weight", type=float, default=None, help="Optional minimum edge weight after plausibility filtering.")
    parser.add_argument("--min-support", type=float, default=0.5, help="Optional minimum bootstrap support when bootstrap_support is available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (args.input_dir / "filtered")
    output_dir.mkdir(parents=True, exist_ok=True)

    edges, source_type, weight_column = read_edges(args.input_dir, prefer_bootstrap=args.prefer_bootstrap)
    thresholded, threshold_rejected = apply_thresholds(edges, weight_column, args.min_weight, args.min_support)
    plausible, plausibility_rejected = apply_plausibility_rules(thresholded)
    filtered, bidirectional_rejected = resolve_bidirectional_edges(plausible, weight_column)

    rejected = pd.concat([threshold_rejected, plausibility_rejected, bidirectional_rejected], ignore_index=True)

    filtered.to_csv(output_dir / "filtered_variable_edges.csv", index=False)
    rejected.to_csv(output_dir / "rejected_variable_edges.csv", index=False)
    save_graph_image(filtered, output_dir / "filtered_variable_graph.png", weight_column=weight_column)
    save_graphml(filtered, output_dir / "filtered_variable_graph.graphml", weight_column=weight_column)

    pd.DataFrame([
        {"key": "input_dir", "value": str(args.input_dir)},
        {"key": "source_type", "value": source_type},
        {"key": "weight_column", "value": weight_column},
        {"key": "min_weight", "value": str(args.min_weight)},
        {"key": "min_support", "value": str(args.min_support)},
        {"key": "n_input_edges", "value": str(len(edges))},
        {"key": "n_filtered_edges", "value": str(len(filtered))},
        {"key": "n_rejected_edges", "value": str(len(rejected))},
    ]).to_csv(output_dir / "filter_metadata.csv", index=False)

    print(f"Filtered graph outputs saved to: {output_dir}")
    print(f"Input edges: {len(edges):,}")
    print(f"Filtered edges: {len(filtered):,}")
    print(f"Rejected edges: {len(rejected):,}")


if __name__ == "__main__":
    main()
