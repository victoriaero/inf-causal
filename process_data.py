from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_INPUT_DIR = BASE_DIR / "data" / "raw" / "sim_selected" / "parquet_by_year_uf"
DEFAULT_CATEGORIES_DIR = BASE_DIR / "categories"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "processed" / "sim_selected"

DEFAULT_OUTPUT_CSV = "dataset.csv"
DEFAULT_AUDIT_CSV = "audit_limpeza_complete_cases.csv"
DEFAULT_MISSING_IMPACT_CSV = "audit_remocao_ignorados_por_variavel.csv"

TIPO_OBITO_COL = "tipo_obito"
IDADE_COL = "idade"
CAUSA_COL = "causa_basica"
ESCOLARIDADE_COL = "escolaridade"

COLUMNS_ALLOWED_TO_IGNORE_FOR_COMPLETE_CASE = {
    "morte_evitavel",
    "escolaridade_grupo",
    "escolaridade_nivel",
    # Variaveis mantidas para EDA/sensibilidade, mas nao usadas para derrubar
    # registros na limpeza complete-case do nucleo analitico.
    "sequencial_obito",
    "hora_obito",
    "naturalidade",
    "assistencia_medica",
    "necropsia",
    "codigo_estabelecimento",
    "atestante",
}

IGNORADO_LIKE = {
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

IGNORADO_CODES = {
    "9",
    "9.0",
    "99",
    "99.0",
    "999",
    "999.0",
    "9999",
    "9999.0",
}

NUMERIC_OR_ID_COLS_NOT_TO_REMOVE_BY_CODE = {
    "ano",
    "idade",
    "sequencial_obito",
    "id_municipio_residencia",
    "id_municipio_ocorrencia",
}

CODE_LIKE_COLUMNS_TO_NORMALIZE = {
    "sequencial_obito",
    "causa_basica",
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
}


def normalize_code_value(value: object) -> str | pd.NA:
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


def normalize_code_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for col in CODE_LIKE_COLUMNS_TO_NORMALIZE:
        if col in df.columns:
            df[col] = df[col].map(normalize_code_value).astype("string")

    return df


def normalize_cid_code(value: object, max_len: int | None = None) -> str:
    """
    Normaliza códigos CID-10 para comparação.

    Exemplos:
    - "I21.9" -> "I219"
    - " i219 " -> "I219"
    - "R99"   -> "R99"
    """
    if pd.isna(value):
        return ""

    code = str(value).strip().upper().replace(".", "")

    if max_len is not None:
        code = code[:max_len]

    return code


def read_code_list(path: Path, max_len: int | None = None) -> set[str]:
    """
    Lê arquivos txt em que cada linha contém um código ou prefixo.
    Linhas vazias e linhas começando com # são ignoradas.
    """
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de categorias não encontrado: {path}")

    codes: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()

            if not raw or raw.startswith("#"):
                continue

            code = normalize_cid_code(raw, max_len=max_len)

            if code:
                codes.add(code)

    return codes


def load_avoidable_categories(categories_dir: Path) -> dict[str, set[str]]:
    return {
        "evitaveis_prefix": read_code_list(categories_dir / "evitaveis_prefix.txt", max_len=3),
        "evitaveis_subcategory": read_code_list(categories_dir / "evitaveis_subcategory.txt", max_len=4),
        "evitaveis_exclude": read_code_list(categories_dir / "evitaveis_exclude.txt", max_len=4),
        "mal_definidas_prefix": read_code_list(categories_dir / "mal_definidas_prefix.txt", max_len=3),
    }


def classify_morte_evitavel(causa_basica: object, categories: dict[str, set[str]]) -> int:
    """
    Classifica mortalidade evitável.

    Códigos:
    - 0 = não evitável
    - 1 = evitável
    - 2 = mal definida

    Prioridade:
    1. mal_definidas_prefix
    2. evitaveis_exclude
    3. evitaveis_subcategory
    4. evitaveis_prefix
    5. demais códigos -> não evitável
    """
    code4 = normalize_cid_code(causa_basica, max_len=4)
    prefix3 = normalize_cid_code(causa_basica, max_len=3)

    if not code4:
        return 2

    if prefix3 in categories["mal_definidas_prefix"]:
        return 2

    if code4 in categories["evitaveis_exclude"]:
        return 0

    if code4 in categories["evitaveis_subcategory"]:
        return 1

    if prefix3 in categories["evitaveis_prefix"]:
        return 1

    return 0

def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip().lower()
    text = " ".join(text.split())
    return text


def recode_escolaridade(value: object) -> str | pd.NA:
    """
    Recodifica escolaridade em três níveis.

    Códigos observados no SIM/Base dos Dados:
    - 0 = Nenhuma
    - 1 = 1 a 3 anos
    - 2 = 4 a 7 anos
    - 3 = 8 a 11 anos
    - 4 = 12 anos e mais
    - 5 = 9 a 11 anos
    - 9 = Ignorado
    """
    v = normalize_text(value)

    baixa = {
        "0",
        "0.0",
        "nenhuma",
        "sem escolaridade",
        "0 anos",

        "1",
        "1.0",
        "1 a 3 anos",
        "1-3 anos",
        "1 a 3",
        "1-3",
    }

    media = {
        "2",
        "2.0",
        "4 a 7 anos",
        "4-7 anos",
        "4 a 7",
        "4-7",

        "3",
        "3.0",
        "8 a 11 anos",
        "8-11 anos",
        "8 a 11",
        "8-11",

        "5",
        "5.0",
        "1 a 8 anos",
        "1-8 anos",
        "1 a 8",
        "1-8",
        "9 a 11 anos",
        "9-11 anos",
        "9 a 11",
        "9-11",
    }

    alta = {
        "4",
        "4.0",
        "12 anos e mais",
        "12 anos ou mais",
        "12 e mais",
        "12 ou mais",
        "12+",
    }

    ignorado = {
        "9",
        "9.0",
        "ignorado",
        "ignorada",
        "ign",
        "ign.",
        "",
        "nan",
        "none",
        "null",
        "<na>",
    }

    if v in ignorado:
        return pd.NA

    if v in baixa:
        return "baixa"

    if v in media:
        return "media"

    if v in alta:
        return "alta"

    return pd.NA


def escolaridade_nivel(value: object) -> int | pd.NA:
    grupo = recode_escolaridade(value)

    if pd.isna(grupo):
        return pd.NA

    return {"baixa": 0, "media": 1, "alta": 2}[grupo]


def is_ignored_or_missing(series: pd.Series, col: str) -> pd.Series:
    technical_missing = series.isna()
    normalized = series.astype("string").str.strip().str.lower()

    text_missing = normalized.isin(IGNORADO_LIKE)

    if col in NUMERIC_OR_ID_COLS_NOT_TO_REMOVE_BY_CODE:
        code_missing = pd.Series(False, index=series.index)
    else:
        code_missing = normalized.isin(IGNORADO_CODES)

    return technical_missing | text_missing | code_missing


def complete_case_mask(df: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=df.index)

    for col in df.columns:
        if col in COLUMNS_ALLOWED_TO_IGNORE_FOR_COMPLETE_CASE:
            continue

        mask = mask | is_ignored_or_missing(df[col], col)

    return ~mask


def ignored_impact_by_variable(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)

    for col in df.columns:
        ignored = is_ignored_or_missing(df[col], col)
        n_ignored = int(ignored.sum())

        rows.append({
                "column": col,
                "n_rows": n,
                "n_ignored_or_missing": n_ignored,
                "pct_ignored_or_missing": n_ignored / n if n else 0.0,
            })

    return pd.DataFrame(rows).sort_values(
        by="n_ignored_or_missing",
        ascending=False,
    )


def list_parquet_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*.parquet"))

    if not files:
        raise FileNotFoundError(f"Nenhum arquivo .parquet encontrado em: {input_dir}")

    return files


def read_parquet_files(files: Iterable[Path]) -> pd.DataFrame:
    frames = []

    for path in files:
        print(f"Lendo: {path}")
        frames.append(pd.read_parquet(path))

    return pd.concat(frames, ignore_index=True)


def validate_required_columns(df: pd.DataFrame) -> None:
    required = {IDADE_COL, CAUSA_COL, ESCOLARIDADE_COL}
    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes nos Parquets: " + ", ".join(missing)
        )


def build_audit_rows(n_original: int, n_after_age: int, n_after_complete_case: int, n_after_recode_escolaridade: int) -> pd.DataFrame:
    rows = [{
            "step": "original",
            "n_rows": n_original,
            "removed_from_previous_step": 0,
            "pct_removed_from_previous_step": 0.0,
            "pct_remaining_from_original": 1.0 if n_original else 0.0,
        },
        {
            "step": "idade_5_74",
            "n_rows": n_after_age,
            "removed_from_previous_step": n_original - n_after_age,
            "pct_removed_from_previous_step": (n_original - n_after_age) / n_original if n_original else 0.0,
            "pct_remaining_from_original": n_after_age / n_original if n_original else 0.0,
        },
        {
            "step": "remove_ignorados_complete_case",
            "n_rows": n_after_complete_case,
            "removed_from_previous_step": n_after_age - n_after_complete_case,
            "pct_removed_from_previous_step": (n_after_age - n_after_complete_case) / n_after_age if n_after_age else 0.0,
            "pct_remaining_from_original": n_after_complete_case / n_original if n_original else 0.0,
        },
        {
            "step": "remove_escolaridade_nao_mapeada",
            "n_rows": n_after_recode_escolaridade,
            "removed_from_previous_step": n_after_complete_case - n_after_recode_escolaridade,
            "pct_removed_from_previous_step": (n_after_complete_case - n_after_recode_escolaridade) / n_after_complete_case if n_after_complete_case else 0.0,
            "pct_remaining_from_original": n_after_recode_escolaridade / n_original if n_original else 0.0,
        },
    ]

    return pd.DataFrame(rows)


def process_data(input_dir: Path, categories_dir: Path, output_dir: Path, output_csv_name: str, audit_csv_name: str, missing_impact_csv_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    categories = load_avoidable_categories(categories_dir)
    files = list_parquet_files(input_dir)

    df = read_parquet_files(files)
    validate_required_columns(df)
    df = normalize_code_columns(df)

    n_original = len(df)
    print(f"\nLinhas originais: {n_original:,}")

    if TIPO_OBITO_COL in df.columns:
        df = df.drop(columns=[TIPO_OBITO_COL])
        print(f"Coluna removida: {TIPO_OBITO_COL}")

    print(df["idade"].describe())

    print(
        df["idade"]
        .value_counts(dropna=False)
        .sort_index()
        .head(30)
    )

    print(
        df["idade"]
        .value_counts(dropna=False)
        .sort_index()
        .tail(30)
    )

    print("Mínimo:", df["idade"].min())
    print("Máximo:", df["idade"].max())
    print("Nulos:", df["idade"].isna().sum())

    print("Idades entre 5 e 74:", df["idade"].between(5, 74, inclusive="both").sum())
    print("Idades fora de 5-74:", (~df["idade"].between(5, 74, inclusive="both")).sum())

    df[IDADE_COL] = pd.to_numeric(df[IDADE_COL], errors="coerce")
    df = df[df[IDADE_COL].between(5, 74, inclusive="both")].copy()
    n_after_age = len(df)
    print(f"Após filtro idade 5-74: {n_after_age:,}")

    df["morte_evitavel"] = df[CAUSA_COL].apply(lambda x: classify_morte_evitavel(x, categories))

    print(df["escolaridade"].value_counts(dropna=False).head(50))
    print(df["escolaridade"].astype(str).str.strip().value_counts(dropna=False).head(50))
    # df["escolaridade"].astype(str).str.strip().value_counts(dropna=False).to_csv("audit_valores_reais_escolaridade.csv")

    missing_impact = ignored_impact_by_variable(df)

    df["escolaridade_grupo"] = df[ESCOLARIDADE_COL].apply(recode_escolaridade)
    df["escolaridade_nivel"] = df[ESCOLARIDADE_COL].apply(escolaridade_nivel).astype("Int64")

    missing_impact_path = output_dir / missing_impact_csv_name
    missing_impact.to_csv(missing_impact_path, index=False)

    mask_complete = complete_case_mask(df)
    df = df[mask_complete].copy()
    n_after_complete_case = len(df)
    print(f"Após remover ignorados/ausentes: {n_after_complete_case:,}")

    df = df[df["escolaridade_grupo"].notna()].copy()
    n_after_recode_escolaridade = len(df)
    print(f"Após remover escolaridade não mapeada: {n_after_recode_escolaridade:,}")

    derived_cols = ["morte_evitavel", "escolaridade_grupo", "escolaridade_nivel"]
    base_cols = [col for col in df.columns if col not in derived_cols]
    df = df[base_cols + derived_cols]

    output_path = output_dir / output_csv_name
    audit_path = output_dir / audit_csv_name

    df.to_csv(output_path, index=False)

    audit = build_audit_rows(n_original=n_original, n_after_age=n_after_age, n_after_complete_case=n_after_complete_case, n_after_recode_escolaridade=n_after_recode_escolaridade)
    audit.to_csv(audit_path, index=False)

    print("\nProcessamento concluído.")
    print(f"CSV tratado: {output_path}")
    print(f"Auditoria da limpeza: {audit_path}")
    print(f"Auditoria de ignorados por variável: {missing_impact_path}")

    print("\nDistribuição de morte_evitavel:")
    print(df["morte_evitavel"].value_counts(dropna=False).sort_index())

    print("\nDistribuição de escolaridade_grupo:")
    print(df["escolaridade_grupo"].value_counts(dropna=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Processa Parquets brutos do SIM e gera CSV tratado complete-case.")

    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Diretório com os Parquets brutos por ano/UF.")
    parser.add_argument("--categories-dir", type=Path, default=DEFAULT_CATEGORIES_DIR, help="Diretório com arquivos txt de categorias de mortalidade evitável.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Diretório de saída.")
    parser.add_argument("--output-csv-name", type=str, default=DEFAULT_OUTPUT_CSV, help="Nome do CSV tratado final.")
    parser.add_argument("--audit-csv-name", type=str, default=DEFAULT_AUDIT_CSV, help="Nome do CSV de auditoria das etapas de limpeza.")
    parser.add_argument("--missing-impact-csv-name", type=str, default=DEFAULT_MISSING_IMPACT_CSV, help="Nome do CSV com impacto de ignorados/ausentes por variável.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    process_data(input_dir=args.input_dir, categories_dir=args.categories_dir, output_dir=args.output_dir, output_csv_name=args.output_csv_name, audit_csv_name=args.audit_csv_name, missing_impact_csv_name=args.missing_impact_csv_name,)

if __name__ == "__main__":
    main()
