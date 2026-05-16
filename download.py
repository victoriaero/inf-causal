from pathlib import Path
from typing import Any
import basedosdados as bd
import pandas as pd

BILLING_PROJECT_ID = "inf-causal"

YEARS = [2012, 2019, 2020, 2021, 2022]

UFS = [
    "AC", "AL", "AP", "AM", "BA", "CE", "DF", "ES", "GO",
    "MA", "MT", "MS", "MG", "PA", "PB", "PR", "PE", "PI",
    "RJ", "RN", "RS", "RO", "RR", "SC", "SP", "SE", "TO",
]

OUT_DIR = Path("data/raw/sim_selected")
PARQUET_DIR = OUT_DIR / "parquet_by_year_uf"
AUDIT_DIR = OUT_DIR / "audit"

PARQUET_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_COUNTS_PATH = AUDIT_DIR / "chunk_counts_year_uf.csv"
SUMMARY_PATH = AUDIT_DIR / "raw_missing_ignored_summary.csv"
TOP_VALUES_PATH = AUDIT_DIR / "raw_top_values.csv"

REQUESTED_COLUMNS = [
    "ano",
    "sigla_uf",
    "sequencial_obito",
    "tipo_obito",
    "causa_basica",
    "data_obito",
    "hora_obito",
    "naturalidade",
    "data_nascimento",
    "idade",
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
]

EMPTY_LIKE_STRINGS = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "na",
    "NA",
    "N/A",
    "None",
    "NONE",
    "null",
    "NULL",
}

IGNORED_LIKE_STRINGS = {
    "ignorado",
    "Ignorado",
    "IGNORADO",
    "ign",
    "IGN",
    "desconhecido",
    "Desconhecido",
    "DESCONHECIDO",
}

SUSPECT_CODES = {
    "0",
    "0.0",
    "9",
    "9.0",
    "99",
    "99.0",
    "999",
    "999.0",
    "9999",
    "9999.0",
    "99900",
}

def get_available_columns() -> pd.DataFrame:
    query = """
    SELECT
      column_name,
      data_type
    FROM `basedosdados.br_ms_sim.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'microdados'
    ORDER BY ordinal_position
    """

    return bd.read_sql(query, billing_project_id=BILLING_PROJECT_ID)


def download_year_uf(year: int, uf: str, columns: list[str]) -> pd.DataFrame:
    cols_sql = ",\n      ".join(columns)

    query = f"""
    SELECT
      {cols_sql}
    FROM `basedosdados.br_ms_sim.microdados`
    WHERE ano = {year}
      AND sigla_uf = '{uf}'
    """

    print(f"Baixando ano={year}, UF={uf}...")
    return bd.read_sql(query, billing_project_id=BILLING_PROJECT_ID)


def convert_dates_to_string(df: pd.DataFrame) -> pd.DataFrame:
    """
    Conversão exclusivamente técnica para evitar problemas de serialização em Parquet
    com tipos DATE/TIME vindos do BigQuery.

    Não altera categorias, não filtra linhas, não substitui ausentes e não trata
    códigos ignorados.
    """
    df = df.copy()

    for col in df.columns:
        col_lower = col.lower()

        if ("data" in col_lower or "hora" in col_lower or str(df[col].dtype) in {"dbdate", "dbtime"}):
            df[col] = df[col].astype("string")

    return df


def normalize_for_audit(value: Any) -> str:
    if pd.isna(value):
        return "<NA>"

    value = str(value).strip()

    if value == "":
        return "<EMPTY>"

    return value


def count_values_in_set(series: pd.Series, values: set[str]) -> int:
    s = series.astype("string").str.strip()
    return int(s.isin(values).sum())


def profile_chunk(df: pd.DataFrame, year: int, uf: str) -> pd.DataFrame:
    """
    Gera diagnóstico bruto por coluna.

    Importante: esta função só conta possíveis ausentes/ignorados/códigos suspeitos.
    Ela não altera o DataFrame salvo em Parquet.
    """
    rows = []

    for col in df.columns:
        series = df[col]
        n = len(series)

        missing_technical = int(series.isna().sum())
        empty_like = count_values_in_set(series, EMPTY_LIKE_STRINGS)
        ignored_like = count_values_in_set(series, IGNORED_LIKE_STRINGS)
        suspect_codes = count_values_in_set(series, SUSPECT_CODES)
        unique_dropna = int(series.dropna().nunique())

        rows.append({
                "ano": year,
                "sigla_uf": uf,
                "column": col,
                "dtype": str(series.dtype),
                "n_rows": n,
                "n_missing_technical": missing_technical,
                "pct_missing_technical": missing_technical / n if n else 0.0,
                "n_empty_like": empty_like,
                "pct_empty_like": empty_like / n if n else 0.0,
                "n_ignored_like": ignored_like,
                "pct_ignored_like": ignored_like / n if n else 0.0,
                "n_suspect_codes": suspect_codes,
                "pct_suspect_codes": suspect_codes / n if n else 0.0,
                "n_unique_dropna": unique_dropna,
            })

    return pd.DataFrame(rows)


def top_values_chunk(df: pd.DataFrame, year: int, uf: str, top_k: int = 20) -> pd.DataFrame:
    rows = []

    for col in df.columns:
        counts = df[col].map(normalize_for_audit).value_counts(dropna=False).head(top_k)

        for value, count in counts.items():
            rows.append({
                    "ano": year,
                    "sigla_uf": uf,
                    "column": col,
                    "value": value,
                    "count": int(count),
                    "pct": float(count / len(df)) if len(df) else 0.0,
                })

    return pd.DataFrame(rows)


def save_or_append_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        return

    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def main() -> None:
    print("Consultando schema da tabela...")

    schema = get_available_columns()
    available = set(schema["column_name"].tolist())

    selected_columns = [col for col in REQUESTED_COLUMNS if col in available]
    missing_from_schema = [col for col in REQUESTED_COLUMNS if col not in available]

    print(f"\nColunas solicitadas: {len(REQUESTED_COLUMNS)}")
    print(f"Colunas encontradas: {len(selected_columns)}")

    if missing_from_schema:
        print("\nColunas não encontradas no schema:")
        for col in missing_from_schema:
            print(f"  - {col}")

    print("\nColunas que serão baixadas:")
    for col in selected_columns:
        print(f"  - {col}")

    chunk_counts = []

    for year in YEARS:
        for uf in UFS:
            out_path = PARQUET_DIR / f"sim_raw_{year}_{uf}.parquet"

            if out_path.exists():
                print(f"Pulando ano={year}, UF={uf}: arquivo já existe.")
                continue

            try:
                df = download_year_uf(year, uf, selected_columns)
                df = convert_dates_to_string(df)
                df.to_parquet(out_path, index=False)
                n_rows = len(df)
                print(f"Salvo: {out_path} | linhas: {n_rows:,}")

                chunk_counts.append({
                        "ano": year,
                        "sigla_uf": uf,
                        "n_rows": n_rows,
                        "path": str(out_path),
                    })

                summary = profile_chunk(df, year, uf)
                top_values = top_values_chunk(df, year, uf, top_k=20)

                save_or_append_csv(summary, SUMMARY_PATH)
                save_or_append_csv(top_values, TOP_VALUES_PATH)

                pd.DataFrame(chunk_counts).to_csv(CHUNK_COUNTS_PATH, index=False)

            except Exception as exc:
                print(f"ERRO em ano={year}, UF={uf}: {exc}")
                print("Continuando para o próximo bloco...")

    print("\nColeta bruta concluída.")
    print(f"Arquivos Parquet em: {PARQUET_DIR}")
    print(f"Auditorias em: {AUDIT_DIR}")


if __name__ == "__main__":
    main()
