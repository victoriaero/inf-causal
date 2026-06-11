# Projeto Causalidade — Dados SIM

Este repositório organiza uma análise de inferência causal com microdados do Sistema de Informações sobre Mortalidade (SIM), com foco na pergunta:

> qual é o efeito causal da escolaridade sobre a ocorrência de mortes evitáveis?

A análise usa um DAG definido previamente para identificar o conjunto de ajuste, exclui causas de morte mal definidas na estimação principal e compara diferentes estimadores para avaliar a robustez dos resultados.

## Estrutura

```text
categories/          Regras auxiliares para classificar causas evitáveis e mal definidas.
causal/              DAG final, validação, estimadores, comparação e sensibilidade.
discovery/      Experimentos de descoberta/filtragem de grafos.
data/                Dados brutos e processados.
dictionaries/        Dicionários e arquivos auxiliares do SIM.
eda/                 Análises exploratórias e figuras.
notebooks/           Protótipos e verificações manuais.
scripts/             Scripts de execução do pipeline.
```

Os notebooks `lucas.ipynb` e `verificacao_refutabilidade.ipynb` ficam em `notebooks/` porque servem como material exploratório e histórico da construção/verificação do DAG. Os códigos canônicos para execução ficam em `causal/`.

## Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

Os comandos abaixo assumem execução a partir da raiz do repositório:

```bash
cd /scratch/victoria.estanislau/inf-causal
```

Se quiser usar um interpretador específico, defina `PYTHON`:

```bash
PYTHON=/caminho/para/python scripts/run_all_experiments.sh estimators
```

## Dados

Para baixar os dados brutos do SIM:

```bash
python3 download.py
```

Os arquivos brutos são salvos em:

```text
data/raw/sim_selected/parquet_by_year_uf/
```

Para processar a base analítica:

```bash
python3 process_data.py
```

A saída principal é:

```text
data/processed/sim_selected/dataset.csv
```

## Variável de desfecho

A variável `morte_evitavel` é criada com três categorias:

```text
0 = não evitável
1 = evitável
2 = mal definida
```

A estimação principal usa apenas `0` e `1`, removendo causas mal definidas. A classificação usa os arquivos em `categories/`.

## Tratamento

A escolaridade é recodificada em três níveis:

```text
baixa
media
alta
```

O tratamento principal é `escolaridade_grupo`.

## DAG e conjunto de ajuste

O DAG final está em:

```text
causal/dag_final.json
```

Para o efeito total de `escolaridade_grupo` sobre `morte_evitavel`, o conjunto de ajuste usado é:

```text
ano, idade_grupo, raca_cor, sexo, sigla_uf
```

Esse conjunto é usado para estimar o efeito total, portanto mediadores como `ocupacao` e `local_ocorrencia` não entram no ajuste principal.

## Execução principal

O script central é:

```bash
scripts/run_all_experiments.sh
```

Ele organiza os resultados em uma pasta única, controlada por `OUT_ROOT`. Exemplo recomendado para uma execução final:

```bash
RUN_DAG=1 OUT_ROOT=causal/output/final_runs/minha_execucao scripts/run_all_experiments.sh all
```

Isso roda:

```text
1. Checagem sintática dos scripts.
2. Validação/refutabilidade do DAG.
3. G-computation.
4. AIPW XGBoost par-a-par com bootstrap.
5. AIPW XGBoost com três classes.
6. Modelo bayesiano hierárquico com três configurações de prior.
7. Comparação entre métodos.
8. Sensibilidade por E-value.
```

Os principais resultados ficam em:

```text
causal/output/final_runs/minha_execucao/
```

## Rodar etapas separadas

Também é possível executar partes do pipeline:

```bash
OUT_ROOT=causal/output/final_runs/minha_execucao scripts/run_all_experiments.sh dag
OUT_ROOT=causal/output/final_runs/minha_execucao scripts/run_all_experiments.sh estimators
OUT_ROOT=causal/output/final_runs/minha_execucao scripts/run_all_experiments.sh bayesian
OUT_ROOT=causal/output/final_runs/minha_execucao scripts/run_all_experiments.sh compare
OUT_ROOT=causal/output/final_runs/minha_execucao scripts/run_all_experiments.sh sensitivity
```

Use o mesmo `OUT_ROOT` para manter os resultados da execução juntos. O arquivo `run_manifest.txt` registra as chamadas feitas.

## Parâmetros úteis

Os principais parâmetros podem ser alterados por variáveis de ambiente:

```bash
BOOTSTRAP_ITERATIONS=300 \
BOOTSTRAP_SAMPLE_SIZE=100000 \
BOOTSTRAP_EVALUATION_SIZE=100000 \
XGB_PAIR_CROSSFIT_FOLDS=3 \
XGB_3CLASS_CROSSFIT_FOLDS=2 \
BAYESIAN_SVI_STEPS=3000 \
BAYESIAN_POSTERIOR_SAMPLES=1000 \
OUT_ROOT=causal/output/final_runs/minha_execucao \
scripts/run_all_experiments.sh all
```

## Resultados principais

Os arquivos mais usados para tabelas e discussão são:

```text
method_comparison/education_effect_methods_risk_difference.csv
method_comparison/education_effect_methods_risk_ratio.csv
aipw_xgb_pair_crossfit3_bootstrap300/aipw_bootstrap_summary_common_support.csv
aipw_xgb_3class_bootstrap300/aipw_3class_bootstrap_summary_global_support.csv
bayesian_sensitivity/bayesian_sensitivity_risk_difference_median.csv
evalue_sensitivity/education_effect_evalues.csv
dag_checks/independence_tests.csv
```

## Observação metodológica

Os resultados devem ser interpretados como evidência compatível com efeito causal sob as hipóteses do DAG, positividade e ausência de confundimento não medido forte. A análise de sensibilidade por E-value ajuda a explicitar o quanto as conclusões poderiam ser afetadas por confundimento não observado.
