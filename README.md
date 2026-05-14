Segue uma versão breve de `README.md`:

````md
# Projeto Causalidade — Dados SIM

Este projeto realiza a coleta, limpeza inicial e preparação de dados do Sistema de Informações sobre Mortalidade (SIM), disponibilizados pela Base dos Dados, para estudos sobre mortalidade evitável no Brasil.

## Estrutura esperada

```text
projeto/
  categories/
    evitaveis_prefix.txt
    evitaveis_subcategory.txt
    evitaveis_exclude.txt
    mal_definidas_prefix.txt

  data/
    raw/
      sim_selected/
        parquet_by_year_uf/
    processed/
      sim_selected/

  coleta_sim_bruta_sem_filtros.py
  processa_sim_tratado.py
  requirements.txt
````

## Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Coleta dos dados brutos

A coleta baixa os dados do SIM por ano e Unidade Federativa, salvando os arquivos em formato Parquet.

O script de coleta mantém os dados brutos sem remover registros ausentes ou ignorados. Apenas conversões técnicas, como datas para string, são feitas para permitir o salvamento adequado em Parquet.

Para executar:

```bash
python3 download.py
```

Os arquivos serão salvos em:

```text
data/raw/sim_selected/parquet_by_year_uf/
```

Também são gerados arquivos de auditoria com contagens, valores ausentes e categorias mais frequentes.

## Seleção inicial de variáveis

A coleta utiliza um subconjunto de colunas do SIM, incluindo informações temporais, geográficas, demográficas, causa básica do óbito e variáveis relacionadas ao registro, como:

* ano
* sigla_uf
* sequencial_obito
* causa_basica
* data_obito
* data_nascimento
* idade
* escolaridade
* sexo
* raca_cor
* estado_civil
* id_municipio_residencia
* id_municipio_ocorrencia
* ocupacao
* local_ocorrencia
* assistencia_medica

A coluna `tipo_obito` é coletada inicialmente, mas removida na etapa de processamento, pois o recorte analítico considera apenas idades entre 5 e 74 anos.

## Processamento dos dados

O script de processamento lê todos os arquivos Parquet brutos, junta os dados e aplica uma limpeza inicial.

Para executar:

```bash
python process_data.py
```

A saída principal é:

```text
data/processed/sim_selected/dataset.csv
```

## Critérios de limpeza e seleção

A limpeza inicial aplica os seguintes critérios:

### 1. Recorte etário

São mantidos apenas registros com idade entre 5 e 74 anos:

```text
5 <= idade <= 74
```

Esse recorte busca evitar dinâmicas específicas de mortalidade infantil e de idades muito avançadas.

### 2. Remoção do ano de 2018

O ano de 2018 é desconsiderado na base tratada. Isso pode ser feito filtrando a coluna `ano` durante o processamento.

### 3. Classificação de morte evitável

É criada a variável `morte_evitavel`, com três valores:

```text
0 = não evitável
1 = evitável
2 = mal definida
```

A classificação usa arquivos auxiliares na pasta `categories/`:

* `evitaveis_prefix.txt`: prefixos CID-10 de três caracteres considerados evitáveis.
* `evitaveis_subcategory.txt`: subcategorias CID-10 específicas consideradas evitáveis.
* `evitaveis_exclude.txt`: subcategorias que devem ser excluídas da lista de evitáveis.
* `mal_definidas_prefix.txt`: prefixos CID-10 considerados causas mal definidas.

A ordem de prioridade é:

```text
1. Causa mal definida
2. Subcategoria explicitamente excluída
3. Subcategoria explicitamente evitável
4. Prefixo evitável
5. Demais causas como não evitáveis
```

### 4. Recodificação da escolaridade

A variável `escolaridade` é recodificada em três níveis:

```text
baixa:
- Nenhuma
- 1 a 3 anos

media:
- 4 a 7 anos
- 8 a 11 anos
- 1 a 8 anos
- 9 a 11 anos

alta:
- 12 anos e mais
```

Nos dados, a escolaridade pode aparecer codificada numericamente. O mapeamento utilizado é:

```text
0 = baixa
1 = baixa
2 = media
3 = media
4 = alta
5 = media
9 = ignorado
```

São criadas duas colunas derivadas:

```text
escolaridade_grupo: baixa, media ou alta
escolaridade_nivel: 0, 1 ou 2
```

### 5. Remoção de registros ignorados ou ausentes

Após a criação das variáveis derivadas, são removidas as linhas com valores ausentes, ignorados ou não informados em qualquer atributo original considerado no modelo.

Essa etapa corresponde a uma análise de casos completos.

Valores como `Ignorado`, `NA`, `None`, `null`, strings vazias e códigos como `9`, `99`, `999` ou `9999` podem ser tratados como ausentes, dependendo da coluna.

## Arquivos de saída

O processamento gera:

```text
data/processed/sim_selected/dataset.csv
```

Base final tratada.

```text
data/processed/sim_selected/audit_limpeza_complete_cases.csv
```

Auditoria com o número de linhas restantes após cada etapa de limpeza.

```text
data/processed/sim_selected/audit_remocao_ignorados_por_variavel.csv
```

Auditoria do impacto de valores ignorados ou ausentes por variável.

```text
data/processed/sim_selected/audit_mapeamento_escolaridade.csv
```

Auditoria opcional para verificar como os valores originais de escolaridade foram recodificados.
