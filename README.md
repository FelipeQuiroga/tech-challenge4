# Tech Challenge — Fase 4: Sustentação e Confiabilidade de Credit Scoring

**Tema central:** Observabilidade Contínua, Detecção de Drift, Qualidade de Dados e Governança em Produção.

Este repositório implementa a camada de **Sustentação e Confiabilidade** para um modelo de Credit Scoring, usando o dataset público [Lending Club](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (empréstimos aceitos, 2007–2018). O objetivo não é maximizar a performance do modelo, e sim demonstrar, de ponta a ponta:

1. **Contratos de dados** que bloqueiam lotes inválidos antes de chegarem ao modelo.
2. **Simulação controlada** de Data Drift e Concept Drift e sua **detecção estatística**.
3. **Observabilidade** (logs, MLflow, dashboard) do pipeline e do modelo em produção.
4. **Governança e LGPD**, com análise causal da degradação.

## Sumário

- [Arquitetura e fluxo](#arquitetura-e-fluxo)
- [Como rodar](#como-rodar)
- [Etapa 1 — Validação de dados e contrato](#etapa-1--validação-de-dados-e-contrato)
- [Etapa 2 — Simulação e detecção de drift](#etapa-2--simulação-e-detecção-de-drift)
- [Etapa 3 — Observabilidade](#etapa-3--observabilidade)
- [Etapa 4 — Governança, LGPD e Causalidade](#etapa-4--governança-lgpd-e-causalidade)
- [Resultados obtidos](#resultados-obtidos)
- [Vídeo STAR](#vídeo-star)

## Arquitetura e fluxo

```
Lending Club (zip Kaggle)
        │  streaming por chunks, sem colunas de PII
        ▼
 src/data_prep.py  ──────────────►  data/processed/{reference,production}.parquet
        │                                  (mesma safra 2015, amostras disjuntas)
        ▼
 src/contracts.py  ──► bloqueia lotes inválidos (Pandera) ──► data/quarantine/
        │
        ▼
 src/train.py  ──────────────────►  models/baseline_model.joblib (XGBoost)
        │
        ▼
 src/simulate_drift.py  ─────────►  6 lotes de produção com drift crescente
        │
        ▼
 src/detect_drift.py  ───────────►  reports/drift_report.html (Evidently)
        │                           reports/drift_metrics.csv (PSI/KS por lote)
        ▼
 src/observability.py  ──────────►  logs/pipeline.jsonl
                                     mlruns/mlflow.db (MLflow)
                                     reports/dashboard.html

 src/run_pipeline.py  orquestra tudo acima, de ponta a ponta.
```

## Como rodar

Pré-requisitos: Python 3.11+, e o zip do dataset Lending Club baixado do Kaggle
(`accepted_2007_to_2018Q4.csv.gz` dentro dele).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows
pip install -r requirements.txt

# aponte para o zip baixado (padrão: ~/Downloads/dataset_att_5.zip)
set LENDING_CLUB_ZIP=C:\caminho\para\o\dataset.zip

python -m pytest tests/ -q          # 18 testes
python -m src.run_pipeline          # pipeline completo ponta a ponta
```

> **Nota (Windows):** em ambientes com Controle de Aplicativo / proteção de
> pastas ativa sobre o diretório sincronizado (ex.: OneDrive), a execução de
> binários nativos do `.venv` pode ser bloqueada. Nesse caso, crie o
> ambiente virtual fora da pasta sincronizada (ele não faz parte do
> versionamento) e aponte para ele ao rodar os comandos acima.

Saídas geradas:
- `reports/drift_report.html` — relatório Evidently (Data Drift Preset).
- `reports/dashboard.html` — dashboard de observabilidade (Plotly).
- `reports/drift_metrics.csv` — PSI/KS/AUC por lote.
- `logs/pipeline.jsonl` — log estruturado de cada etapa do pipeline.
- `mlruns/mlflow.db` — rastreamento MLflow (`mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`).
- `data/quarantine/` — lotes rejeitados pelo contrato, com o detalhe dos erros.

## Etapa 1 — Validação de dados e contrato

**Objetivo:** garantir que "lixo não entre no modelo".

- **Features usadas** (sem nenhuma coluna de identificação/PII): `loan_amnt`,
  `term`, `int_rate`, `installment`, `emp_length`, `home_ownership`,
  `annual_inc`, `verification_status`, `purpose`, `dti`, `delinq_2yrs`,
  `fico_range_low`, `open_acc`, `pub_rec`, `revol_bal`, `revol_util`,
  `total_acc`. Target: `loan_status` ∈ {Fully Paid=0, Charged Off=1}.
- **Contrato de dados** (`src/contracts.py`, Pandera): 17 colunas com regras
  de nulidade, range de negócio e domínio, mais uma checagem de duplicatas —
  bem acima do mínimo de 3 regras exigido. Exemplos: `annual_inc` ∈
  (0,01, 500.000] e não nulo; `dti` ∈ [0, 100]; `term` ∈ {36, 60};
  `fico_range_low` ∈ [300, 850]; `home_ownership`/`verification_status` em
  domínio fechado.
- **Demonstração do bloqueio:** `contracts.make_dirty_batch()` gera um lote
  sintético com nulos, renda negativa, FICO absurdo, `term` inválido e
  duplicatas. `enforce_contract()` rejeita esse lote (`ValueError`), grava o
  lote e o detalhe das violações em `data/quarantine/` e loga o evento —
  reproduzido automaticamente em `run_pipeline.py` e coberto por
  `tests/test_contracts.py` (9 testes).
- **Baseline:** Pipeline scikit-learn (imputação + one-hot) + `XGBClassifier`
  treinado sobre o Dataset de Referência já validado pelo contrato.

## Etapa 2 — Simulação e detecção de drift

**Separação Referência vs. Produção:** para isolar o efeito do drift
*intencional* do drift *real* que já existe entre safras diferentes da
Lending Club (mudanças de política de crédito ao longo dos anos), referência
e produção-base são amostras **disjuntas da mesma safra (2015)**. A passagem
do tempo em produção é simulada fatiando essa base em 6 lotes sequenciais e
aplicando uma perturbação **crescente** (`src/simulate_drift.py`):

- **Data Drift:** `annual_inc` sobe até +35% (inflação nominal), `dti` sobe
  até +50% (endividamento), `int_rate` sobe até +6 p.p. (custo de crédito).
- **Concept Drift:** empréstimos antes considerados "bons" passam a virar
  default com probabilidade crescente conforme o perfil de risco (dti alto,
  renda relativa baixa) — ou seja, a mesma entrada passa a ser decidida de
  forma diferente, não só a distribuição das features muda.
- No lote 0 (intensidade zero) o lote é um passe neutro exato, servindo de
  controle/sanidade: a detecção de drift não acusa nenhuma feature alterada.

**Detecção estatística** (`src/detect_drift.py`, Evidently AI):
- **PSI** é o critério **primário** de alerta (numéricas e categóricas),
  por ser uma métrica de *effect size* pouco sensível ao tamanho da amostra.
  Testes de p-valor (KS, qui-quadrado/Z) foram descartados como critério de
  decisão porque, com os tamanhos de amostra usados aqui (60k referência vs.
  10k por lote), acusam "significância estatística" até para diferenças
  puramente de ruído amostral — ver [Armadilhas estatísticas](#armadilhas-estatísticas-p-valor-vs-effect-size).
- O **p-valor de KS** ainda é calculado e reportado para as features
  numéricas (citado explicitamente no enunciado), de forma informativa.
- `reports/drift_report.html`: relatório HTML do Evidently (Data Drift
  Preset) comparando referência × lote mais degradado.
- `reports/drift_metrics.csv`: PSI, KS, AUC e taxa de default por lote.

## Etapa 3 — Observabilidade

`src/observability.py` centraliza:

- **Logs estruturados** (`logs/pipeline.jsonl`): um evento JSON por etapa
  (ingestão de dados, validação de contrato, treino, análise de drift),
  contendo `status`, `latência`, volume processado e contadores de alerta.
- **MLflow** (backend SQLite local em `mlruns/mlflow.db`, sem servidor
  externo): um run para o treino do baseline (params + AUC/KS) e um run por
  lote de produção analisado (AUC, queda de AUC, nº de features com drift,
  features afetadas como tag).
- **Dashboard HTML** (`reports/dashboard.html`, Plotly): evolução da AUC vs.
  a referência, taxa de default observada, nº de features com drift por
  lote e contagem de eventos do pipeline por status.

**Métricas de saúde operacional documentadas:**

| Métrica | Onde | Uso |
|---|---|---|
| Status de cada etapa (success/alert/error) | `pipeline.jsonl`, MLflow tags | detectar falhas de ingestão/validação |
| Latência por etapa (s) | `pipeline.jsonl` | identificar gargalos no pipeline |
| Volume de linhas processadas | `pipeline.jsonl` | detectar quedas anômalas de volume |
| AUC do modelo por lote vs. referência | `drift_metrics.csv`, MLflow, dashboard | degradação de performance (concept drift) |
| PSI / KS por feature e por lote | `drift_metrics.csv`, `drift_report.html` | degradação de distribuição (data drift) |
| Nº de alertas de drift/falha | `pipeline.jsonl`, dashboard | visão agregada de saúde |

## Etapa 4 — Governança, LGPD e Causalidade

### Privacidade e LGPD (Privacy by Design / by Default)

- **Minimização de dados (Privacy by Design):** apenas 16 colunas
  financeiras/comportamentais são lidas do dataset bruto (ver Etapa 1).
  Nenhuma coluna de identificação direta ou indireta é carregada em nenhum
  momento do pipeline — `id`, `member_id`, `zip_code`, `addr_state`,
  `emp_title`, `desc`, `url`, `title` nunca entram em memória
  (`src/config.py::ALL_USECOLS`). Isso é reforçado por `usecols` na própria
  leitura do CSV, não por um filtro posterior.
- **Base legal (hipotética, para um cenário real de produção):** a decisão
  de concessão/precificação de crédito se apoia no art. 7º, V (execução de
  contrato/procedimentos preliminares a pedido do titular) e, para
  variáveis de enriquecimento/modelagem de risco, no art. 7º, IX (legítimo
  interesse), sempre sujeito a teste de proporcionalidade e balanceamento
  com os direitos do titular (art. 10).
- **Decisão automatizada (art. 20):** por se tratar de decisão que afeta os
  interesses do titular, o processo real precisa prever (a) direito a
  solicitar revisão por pessoa natural e (b) explicação clara e adequada
  dos critérios da decisão — aqui viabilizado pela natureza interpretável
  do baseline (feature importance do XGBoost / `installment`, `dti`,
  `int_rate` como principais drivers) e pelos relatórios de drift, que
  tornam auditável *quando* e *por quê* o comportamento do modelo mudou.
- **Retenção e descarte:** dados de treino/produção (parquet) e logs ficam
  fora do controle de versão (`.gitignore`) e, num cenário real, seguiriam
  uma política de retenção vinculada à finalidade (ex.: prazo legal de
  guarda de operações de crédito, hoje de até 5 anos após o encerramento da
  relação, revisável conforme regulação setorial vigente), com anonimização
  ou purga ao fim do prazo.
- **Privacy by Default:** o pipeline só processa o conjunto mínimo de
  colunas por padrão (`FEATURE_COLUMNS` em `config.py`); qualquer expansão
  do escopo de dados exigiria alteração explícita de código, não é o
  comportamento padrão.

### Mitigação de viés

- Variáveis de alta sensibilidade/potencial proxy de atributos protegidos
  (`zip_code`, `addr_state`, `emp_title`) são excluídas desde a ingestão,
  não apenas descartadas depois — evitando que o modelo aprenda
  correlações geográficas/profissionais como proxy indireto de raça, renda
  regional ou outras características protegidas.
- O relatório de drift é segmentável por `home_ownership`/`purpose` (não
  feito neste MVP, mas suportado pela estrutura de `detect_drift.py`) para
  checar se a degradação atinge desproporcionalmente algum subgrupo.

### Análise causal da degradação

```
Inflação / mudança macroeconômica
        │
        ▼
Renda nominal sobe, mas poder de compra real não acompanha
        │
        ▼
Razão dívida/renda (dti) e custo do crédito (int_rate) sobem
        │
        ▼
Perfis antes "bons" tornam-se, de fato, mais arriscados (Concept Drift)
        │
        ▼
O modelo, treinado sob a relação antiga feature→risco, passa a
SUBESTIMAR o risco desses perfis → AUC cai, inadimplência observada sobe
```

Essa cadeia é exatamente o que `simulate_drift.py` reproduz de forma
controlada: **Data Drift** (mudança na distribuição de `annual_inc`, `dti`,
`int_rate`) **causa** parte do problema, mas o fator dominante é o
**Concept Drift** (a relação entre essas variáveis e o risco real de
default muda) — por isso a AUC cai mesmo quando apenas 1–3 features são
sinalizadas estatisticamente como "drift de dados" (lotes 1–2), e continua
caindo nos lotes seguintes.

### Armadilhas estatísticas: p-valor vs. effect size

Com `n` grande (60.000 na referência vs. 10.000 por lote), testes de
hipótese como KS ou qui-quadrado tendem a rejeitar a hipótese nula
("as distribuições são iguais") mesmo para diferenças irrelevantes na
prática, porque o poder estatístico do teste cresce com `n`. Por isso este
projeto usa **PSI** (uma métrica de magnitude de mudança, não de
significância) como critério de decisão, documentando aqui que
"estatisticamente significativo" não é o mesmo que "accionável" — e que
**correlação não implica causalidade**: a queda de AUC coincide com o
drift simulado porque o experimento foi desenhado para isso, não porque
qualquer associação observada em produção real necessariamente teria a
mesma causa.

## Resultados obtidos

Execução de referência (`reports/drift_metrics.csv`), baseline com AUC de
referência ≈ **0,733** (KS ≈ 0,341):

| Lote | AUC | Queda vs. referência | Taxa de default | Features com drift (PSI) | Alerta |
|---|---|---|---|---|---|
| 0 | 0,722 | 0,012 | 19,6% | — (0) | não |
| 1 | 0,708 | 0,025 | 22,9% | `int_rate` | não |
| 2 | 0,688 | 0,046 | 25,6% | `dti`, `int_rate` | **sim** |
| 3 | 0,687 | 0,046 | 27,9% | `dti`, `int_rate` | **sim** |
| 4 | 0,670 | 0,064 | 30,2% | `annual_inc`, `dti`, `int_rate` | **sim** |
| 5 | 0,662 | 0,071 | 33,5% | `annual_inc`, `dti`, `int_rate` | **sim** |

O lote 0 (sem drift sintético) confirma a ausência de falsos positivos; a
partir do lote 2, o sistema de observabilidade gera alerta de degradação de
performance de forma consistente e crescente.

## Vídeo STAR

_Link a ser adicionado após a gravação._

---

## Estrutura do repositório

```
src/
  config.py          caminhos, colunas, limiares — configuração central
  data_prep.py        ingestão streaming do Lending Club (Etapa 1)
  contracts.py         contrato de dados Pandera (Etapa 1)
  train.py             baseline XGBoost (Etapa 1)
  simulate_drift.py    simulação de Data/Concept Drift (Etapa 2)
  detect_drift.py       detecção PSI/KS com Evidently (Etapa 2)
  observability.py       logs, MLflow, dashboard (Etapa 3)
  run_pipeline.py          orquestra Etapas 1–3 ponta a ponta
tests/                      18 testes automatizados (pytest)
reports/                     relatórios HTML/CSV gerados (versionados)
data/, models/, logs/, mlruns/   artefatos gerados (fora do versionamento)
```
