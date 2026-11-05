# World Bank Procurement Multi-Agent Demo

This folder is the open-source backend companion to `WorldBank_MultiAgent_Demo.py`.
It makes the requested external-technology mapping executable rather than treating
it as an architecture-only note. The original notebook remains available for the
Databricks-native Workspace, SQL, Delta/Unity Catalog, MLflow, Jobs, and system-table
walkthrough.

## What the uploaded notebook does and does not do

The notebook is a useful Databricks prototype: it calls the World Bank indicator
API, writes Delta tables, runs SQL, creates basic entity links and risk rules,
logs application requests, and sketches MLflow evaluation/model packaging.
However, it is not a complete implementation of the requested system:

- The procurement and lesson rows are a handful of hard-coded illustrative examples,
  not a labeled historical procurement feed.
- PDF/DOCX parsing and embeddings are described but not run; retrieval is keyword
  `LIKE` matching, not semantic/vector retrieval.
- Its procurement “risk score” is a transparent rule, not a trained timeline,
  amendment, or rebid model.
- The displayed workflow cell describes a job but does not create one. Serving
  endpoint creation is also left to a UI/manual step.
- The source has repeated agent/evaluation definitions and account-specific paths;
  its final cells depend on pre-existing workspace models and tables.
- It does not implement the OSS technology map in the prompt.

This companion supplies the OSS implementation. The sample procurement CSV and
sample ICR/ISR-like text documents are explicitly **fictional test fixtures**,
not World Bank contract records or publications. World Bank macro/governance
indicators are fetched from the public API; actual procurement model training
requires an independently sourced, reviewed, labeled procurement feed.

The optional AI procurement contracts workbook is treated as a separate
structured dataset, not as a document for semantic retrieval or as training data
for the delay/amendment/rebid models. Validate it and create annual counts with:

```powershell
python -m worldbank_demo_backend.ai_procurement
```

By default this reads `examples/AI_Procurement_Contracts.xlsx` and writes
`data/processed/ai_procurement_annual_counts.csv` plus
`data/processed/ai_procurement_validation.json`. It also writes
`data/processed/ai_procurement_source_year_summary.csv` and
`data/processed/ai_procurement_baseline_backtest.json`. The count table groups
unique, dated notices by year, source, income group, and tool category. Missing
dates are excluded and reported; unknown group/category values remain visible as
`Unknown`. The workbook's dates use month/day/year (confirmed by records where
the second date component exceeds 12).

The source-year summary makes observed coverage explicit. Missing source-years
are not interpreted as zero. The baseline backtest evaluates each source
independently over up to five recent contiguous years using expanding-window,
one-year-ahead origins. Sources with too little history are marked not evaluated.
This dataset-level backtest is descriptive: source coverage may be incomplete or
change over time, so it must not be interpreted as a forecast of all real-world
procurement.

An interactive walkthrough is available in
[`AI_Procurement_Forecasting.ipynb`](./AI_Procurement_Forecasting.ipynb). Run it
from the project root using the Python 3.11/3.12 environment with dependencies
from `requirements.txt`.

To run the same exploratory FLAML time-series forecasts for Open Contract and
World Bank counts from the command line, compare them against the same
multi-origin baselines, and log the run to MLflow:

```powershell
$env:MLFLOW_TRACKING_URI = "sqlite:///data/processed/mlflow.db"
python -m worldbank_demo_backend.ai_procurement --train-flaml --flaml-time-budget-seconds 10
```

Set `MLFLOW_TRACKING_URI` to an MLflow OSS tracking server for shared tracking.
TED is intentionally excluded because only one year is present. The run is tagged
`provenance_status=unverified` and `model_intended_use=exploratory_only_not_for_decisions`;
it logs but never registers the model. The JSON forecast report is written to
`data/processed/ai_procurement_flaml_forecasts.json`. Use Python 3.11 or 3.12
with dependencies installed from `requirements.txt`; the forecast extras install
FLAML's time-series estimators.
The report labels the final value as a one-step estimate beyond the latest
observed workbook year and states whether that target is historical, in the
current calendar year, or in the future. If the workbook ends in 2025, the 2026
value is not a validated full-year 2026 forecast because the workbook has no
2026 actuals. Treat any estimate as exploratory, not a basis for decisions.

## Architecture and technology mapping

| Capability | OSS implementation | Evidence/artifact |
|---|---|---|
| Workspace / lakehouse flow | Databricks notebook plus this deployable project | notebook and API |
| Spark / ETL | PySpark | `pipeline.py` Bronze/Silver/Gold tasks |
| ACID storage | Delta Lake OSS | idempotent Delta `MERGE` by stable keys |
| Catalog / governance | Hive Metastore or Project Nessie; API role/project authorization | `governance.py`, `api.py` |
| Experiment tracking / registry | MLflow | `modeling.py`, evaluation metrics and candidate models |
| Automated model selection | FLAML | timeline regression and amendment/rebid classifiers |
| Data and model-output drift | Evidently AI | `monitoring.py`, Airflow report task |
| Multi-agent orchestration | LangGraph | risk, evidence, evidence adjudicator, faithfulness guard |
| Inference serving | FastAPI + Docker | `api.py`, `Dockerfile` |
| Scheduled workflows | Apache Airflow | `airflow/dags/worldbank_pipeline.py` |
| SQL analytics | DuckDB | parameterized project lookup and sector aggregates |
| Logs/traces | JSON application/audit log + OpenTelemetry OTLP | `observability.py` |

Unity Catalog itself is a Databricks service and is not emulated by OSS components.
In an OSS deployment, Nessie/Hive catalog metadata plus the backend's explicit
role/project checks demonstrate governance concepts; configure actual object-store
ACLs and secret access in the deployment platform.

For Delta tables, use the Hive catalog path. This project also exposes a Nessie
Iceberg catalog when configured, but Delta tables are currently path-addressed and
are not registered in Nessie. The local end-to-end smoke check below validates
Delta Lake with Hive; it does not claim that Nessie catalogs Delta tables.

## Local Docker and Airflow validation

The local Compose stack provides an API backed by the clearly synthetic example
feed/documents, an Airflow scheduler, and a manual Delta Lake/Hive validation DAG.
It is for development smoke testing only—not production data, security, sizing, or
availability validation. Docker Desktop with the Compose plugin is required.

Build and start the API and Airflow:

```powershell
docker compose config
docker compose build
docker compose up -d api airflow
```

Check the API and open Airflow at `http://127.0.0.1:8080`:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
docker compose logs airflow
```

Airflow prints the generated local login credentials in its startup logs.
Trigger the `worldbank_local_stack_smoke` DAG from the Airflow UI. It writes and
merges a temporary Delta table, registers it in the local Hive catalog, reads it
back, and fails if the values do not match. The regular procurement workflow DAG
is created paused by default; configure reviewed feeds, credentials, and
evaluation/monitoring inputs before unpausing it. That workflow can register
candidate models when new data is detected, so verify its training/registration
policy before enabling its schedule.

You can run the storage check without starting Airflow:

```powershell
docker compose --profile smoke run --rm delta-smoke
```

The local Hive smoke test uses Spark's embedded metastore unless
`HIVE_METASTORE_URIS` is configured. For a shared/external deployment, set that
variable to the intended Hive Metastore Thrift URI and validate storage credentials
and ACLs separately. For local Compose, the embedded metastore database and Spark
warehouse are kept under the ignored `data/` directory. The API's default demo
bearer tokens and sample records are not production credentials/data. Replace them
and provide a reviewed procurement feed before any operational use. The API
downloads its sentence-transformer model into a persistent local cache on first
start.

The forecasting notebook is configured for the project `.venv` Python 3.12
interpreter. The workspace setting and registered Jupyter kernelspec are
`worldbank-demo-py312`; select **Python 3.12 (World Bank demo .venv)** in VS Code's
notebook kernel picker. The environment is expected at `.venv\Scripts\python.exe`.

## Deploy the demo to Render

The root `render.yaml` Blueprint deploys the FastAPI app and its same-origin Q&A
page as one Python web service. Connect a GitHub repository containing this project
to Render and create a Blueprint from that repository. The web service is set to
Render's free plan for an initial demo attempt; free services can sleep when idle,
have limited resources, and may not have enough memory to load the
sentence-transformer model. Upgrade the service plan if it fails to start or
cannot keep the model loaded.

Before creating the service, generate a unique API token locally in PowerShell:

```powershell
$bytes = [byte[]]::new(32)
[Security.Cryptography.RandomNumberGenerator]::Fill($bytes)
$token = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+', '-').Replace('/', '_')
$apiTokens = @{ $token = @{ subject = "demo-owner"; roles = @("admin"); project_ids = @("*") } } | ConvertTo-Json -Compress
$apiTokens
```

In the Render Blueprint setup, provide the printed JSON as the secret value for
`API_TOKENS_JSON`. Do not commit the generated token or share it publicly. After
deployment, open the service URL and paste the token into the Q&A page. The bundled
procurement feed and documents are still fictional demo fixtures; do not treat
their output as real project analysis. This initial configuration is for a
demonstration, not a production-secured deployment.

The service uses an ephemeral filesystem on the free plan. Application logs and
feedback written by the app are not durable there, and the Hugging Face model cache
may need to be downloaded again after a restart. No persistent disk, scheduled
workflow, Delta table, Airflow scheduler, or Nessie service is provisioned by this
Blueprint. Configure and validate those separately before treating this as a
production pipeline.

## Data pipeline and knowledge model

1. **Bronze:** paginated World Bank GDP growth, CPI, population, and WGI indicators
   are retained with source URLs and country/indicator/year keys.
2. **Silver:** a CSV/JSON contract/procurement feed is validated and enriched with
   the latest available not-after-project-year government-effectiveness,
   regulatory-quality, and control-of-corruption observations (within two years,
   preventing future-indicator leakage). PDF, DOCX, and TXT
   documents are parsed with `pdfplumber`/`python-docx` and their tables are kept
   as row-preserving chunks. A JSON manifest supplies reviewed project IDs, sector
   codes, document type, dates, source identifiers, and optional claim annotations.
3. **Gold:** traceable historical risk signals and their source record IDs are
   written as Delta. They are labeled explicitly as heuristic signals, not model
   predictions.
4. **Entity resolution:** exact project ID has confidence 1.0. Where that key is
   missing, matching uses normalized sector plus a configurable temporal window;
   those links retain a lower confidence and year-distance field. Both links and
   source URIs are stored.
5. **Retrieval:** section-aware narrative chunks use 512-word windows with 64-word
   overlap. Table rows remain intact, table headers are repeated, and long tables
   are grouped under the same word-count budget. Sentence-transformer embeddings are
   stored beside the Delta chunks and ranked by cosine similarity for retrieval.
   OCR is not silently assumed: scanned PDFs need an OCR preprocessing step.

The graph compares ICR/ISR evidence only when reviewed annotations identify the
same project and claim-period key. ICRs are terminal retrospectives and ISRs are
time-scoped supervision records; the system does not globally rank one institution
source above another. Incompatible values for the same annotated period are both
returned and explicitly escalated for human review.

## Modeling and guardrails

`modeling.py` uses FLAML to fit (a) actual completion days, (b) any-amendment
classification, and (c) rebid classification. It uses pre-award fields only,
explicitly excludes the realized targets and post-award fields, and uses a
chronological holdout when possible. MLflow records validation metrics including
MAE, precision/recall, ROC AUC, and Brier score, and stores candidate PyFunc models.

Training requires at least 100 labeled records and the required columns. Five
synthetic demo rows intentionally fail that gate. Model outputs are decision
support, not a procurement decision; probability scores are not automatically
calibrated or promoted. Validate calibration, subgroup performance, leakage,
thresholds, and procurement policy with subject-matter reviewers before use.

The API produces deterministic source excerpts and typed claims rather than
free-form unsupported recommendations. Risk/forecast claims carry a procurement
record or model citation; retrieved passages carry document/chunk citations.
Uncited or unretrieved citations block the response. Empty retrieval abstains.
The model does not autonomously award, reject, or rank suppliers.

## Run locally (Windows PowerShell)

Python 3.11 and Java 17 are required for local Spark. Model training and API
embedding also require downloading the selected sentence-transformer model.

```powershell
Set-Location C:\Users\praka\Downloads\worldbank_demo_backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

The `.env` defaults point to synthetic examples, so they are suitable only for
testing parser/retrieval behavior. The API requires a configured bearer token and
starts only after at least one source feed or document collection is available.

Run the public-source ETL after replacing the synthetic contract CSV with a
reviewed feed:

```powershell
python -m worldbank_demo_backend.pipeline
```

The pipeline writes local Delta tables below `./data/lakehouse` by default.
For serving from Delta after the pipeline has created its tables, set
`DELTA_LAKE_ROOT=./data/lakehouse`; the API refreshes its procurement and
evidence snapshots when Delta versions change. Without that setting, it reads
the configured source feed and manifest directly. For multi-worker Airflow
and serving, set the variable to shared storage (for example `s3a://...` for
Spark; the API's Delta-RS reader maps `s3a://` to `s3://`). Set
`DELTA_STORAGE_OPTIONS_JSON` through a secret manager for object-store access.
Never put cloud credentials in this file.

Train and log candidate models only after supplying at least 100 valid labeled
procurement records:

```powershell
python -m worldbank_demo_backend.modeling .\data\lakehouse\silver\procurement_records
```

Add `--register` only after review. Set the same MLflow tracking/registry
configuration for training and serving. Copy the reported registered versions to
the `MLFLOW_MODEL_*_URI` environment variables, restart the API, and validate the
candidate against the human-reviewed evaluation set.

Start the API:

```powershell
uvicorn worldbank_demo_backend.api:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000/` for the local Procurement Q&A page. It sends
requests to the same API origin, displays returned claims and evidence citations,
and labels the bundled synthetic data as fictional. Enter the configured bearer
token in the page (the local demo token is prefilled). The API documentation
remains available at `http://127.0.0.1:8000/docs`.

Example existing-project query:

```powershell
$headers = @{ Authorization = "Bearer demo-admin" }
$body = @{
  project_id = "P005"
  question = "What procurement lesson should be checked for a new water package?"
} | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:8000/v1/query -Method Post `
  -Headers $headers -ContentType "application/json" -Body $body
```

For a new package, send `package_context` with `country_code`, `sector_code`,
`project_year`, and `planned_days` (plus optional procurement method and WGI
scores). It retrieves sector/time-linked precedents even when the new project ID
does not exist in historical procurement records. Forecast scores are returned
only when registered models are configured; otherwise risk prediction abstains.

The sample admin token in `.env.example` is public demo text. Replace it before
any network exposure. The API role map is a local demonstration; production
should validate OIDC/JWT identity and enforce matching cloud data/object-store ACLs.

## Workflows, evaluation, monitoring, and feedback

- For the local Compose scheduler, the pinned Airflow image installs the runtime
  requirements using Airflow's Python 3.11 constraints. For a separate deployment,
  install this project and compatible dependencies on every worker, following
  Airflow's constraints for its chosen Python version; do not install Airflow from
  the runtime requirements file.
- Configure `DRIFT_REFERENCE_DATA_PATH` with reviewed baseline procurement
  features and `PROCUREMENT_DATA_PATH` with the current feed. Evidently reports
  data drift. The API stores model-output snapshots at `PREDICTION_LOG_PATH`;
  compare reviewed reference and current prediction JSONL files with
  `MODEL_DRIFT_REFERENCE_DATA_PATH` and `MODEL_DRIFT_CURRENT_DATA_PATH`.
- Configure `EVAL_CASES_PATH` with human-reviewed expected statuses, relevant
  document IDs, and citations. Evaluation logs retrieval precision/recall/F1,
  citation recall, and abstention/status accuracy to MLflow.
- New contract/milestone feed rows are idempotently merged by contract ID. New or
  revised documents are reparsed and re-embedded, then merged by deterministic
  chunk ID. Airflow retrains candidates only when the Delta content/feed digest
  changes; registration does not automatically make a model active.
- Set `APPROVED_CONTRACT_UPDATE_FEEDS` to comma-separated CSV/JSON contract
  snapshot feeds to ingest reviewed milestones or amendments. Each supplemental
  row must contain `review_status=approved`, `reviewed_by`, an ISO-8601
  timezone-aware `reviewed_at`, a source URI, and a stable contract/tender ID;
  ambiguous duplicate snapshots stop ingestion. The service never converts a
  user rating into a training label automatically.
- API ratings and reviewer comments are persisted to `FEEDBACK_JSONL_PATH` and
  tagged as requiring review. They are not used as labels until a reviewer
  verifies and promotes the corresponding milestone/contract fact into a trusted
  source feed or claim manifest.
- JSON application/audit events go to stdout and rotating `APP_LOG_PATH`.
  Configure `OTEL_EXPORTER_OTLP_ENDPOINT` to export traces/metrics to an
  OpenTelemetry Collector and Grafana-compatible backend. Mount log/artifact
  directories in Docker so logs survive container replacement.

## Pilot rollout and measurable KPIs

1. **Data readiness (weeks 1-2):** agree on procurement keys, document manifests,
   access policies, and labeled outcome definitions; measure completeness and
   entity-link precision against a human-reviewed sample.
2. **Shadow evaluation (weeks 3-6):** run retrieval and risk candidates without
   affecting decisions; review source attribution, temporal leakage, and
   subgroup error; establish a baseline for planning cycle time.
3. **Limited pilot (weeks 7-10):** enable the authorized procurement team with
   human approval required; collect feedback and audit every query/model version.
4. **Scale decision:** promote only reviewed model versions and documents; retain
   rollback, appeal, and access-review procedures.

Recommended pre-registered pilot KPIs (measure before and after; these are targets,
not results): procurement planning median cycle time (target at least 15% lower),
retrieval precision@5 (at least 0.80), retrieval recall@5 (at least 0.75),
supported-claim citation coverage (100%), abstention accuracy on unanswerable
cases (at least 0.95), amendment/rebid Brier score and calibration by country/
sector, model drift, human override rate, and monthly active procurement users.

## Tests

```powershell
python -m unittest discover -s tests -v
```
