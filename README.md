# Customer Segmentation & Classification - Credit Card Customer Data

## Feature Stories 1-3: run the completed workflow

**Added workflow:** [`notebooks/stories_1_to_3.py`](notebooks/stories_1_to_3.py) runs EDA, cleaning, PCA, K-means, DBSCAN, and final cluster selection from the repository dataset. The updated [`notebooks/CardProfiler_cleaning.ipynb`](notebooks/CardProfiler_cleaning.ipynb) covers Stories 2 and 3 with small, explained cells and saved outputs; Story 1 stays in `CardProfiler_EDA_repport.ipynb`. Comments beginning with `ADDED` identify the new notebook work. The earlier cleaning notebook is preserved as `CardProfiler_cleaning_before_update.ipynb`.

From the project root:

```bash
uv sync
uv run python notebooks/stories_1_to_3.py
```

The script reads `data/raw/dataset.csv`. It writes the cleaned data, clustering matrix, search tables, and decision summary to `data/processed/`. It writes nine figures to `repport/stories_1_to_3/`.

**How the final choice is made:** K-means candidates must have no cluster below 10% or above 90%. The candidate with the highest sampled silhouette among those is selected. DBSCAN replaces it only if DBSCAN assigns every customer to a cluster, has the same size balance, and improves silhouette by at least 0.05. Read `data/processed/clustering_summary.txt` for the values from your run. The silhouette calculation uses a fixed sample of up to 1,000 customers for practical run time; DBSCAN's score excludes noise and its noise percentage is shown separately.

This workflow stops at Feature Story 3. Business segment names and the supervised `target` belong to Feature Story 4.

## Overview

This project builds an end-to-end machine learning workflow for **customer segmentation and segment prediction** from credit-card usage behavior.

The project has two complementary objectives:

1. **Unsupervised learning:** discover homogeneous customer behavior profiles using clustering.
2. **Supervised learning:** train a classifier able to predict the final business segment of a new customer.

The workflow covers exploratory data analysis, preprocessing, PCA, K-means and DBSCAN clustering, cluster interpretation, supervised classification, experiment tracking with MLflow, and a Streamlit demonstration application. Apache Airflow orchestration is treated as an optional bonus layer.

---

## Business Context

A financial institution wants to better understand its active credit-card customers from six months of behavioral data. The dataset contains approximately **9,000 active customers** described by seven behavioral variables plus a unique customer identifier.

The expected outcome is not only a clustering notebook, but a reproducible machine-learning project where discovered clusters are converted into understandable business segments and then predicted automatically for new customer profiles.

---

## Dataset

| Variable | Description |
|---|---|
| `CUST_ID` | Unique customer identifier |
| `BALANCE` | Remaining account/card balance available for purchases |
| `PURCHASES` | Total amount of purchases |
| `ONEOFF_PURCHASES` | Total amount of one-off purchases |
| `INSTALLMENTS_PURCHASES` | Total amount of installment purchases |
| `CASH_ADVANCE` | Total cash advance amount |
| `CREDIT_LIMIT` | Credit limit granted to the customer |
| `PAYMENTS` | Total payments made during the observed period |

> `CUST_ID` is an identifier and must not be used as a behavioral feature for modeling.

---

## Project Objectives

### 1. Exploratory Data Analysis

- Inspect dataset shape, types, descriptive statistics, and sample rows.
- Detect missing values and duplicates.
- Measure skewness of numerical variables.
- Visualize numerical distributions with histogram + KDE.
- Study feature correlations with a heatmap.
- Detect potential outliers with boxplots and IQR-based inspection.
- Produce a sorted missing-value summary containing counts and percentages.

### 2. Data Preprocessing

The project intentionally maintains **two separate data branches**.

#### Reference dataset: `df_clean`

`df_clean` is the cleaned reference dataset used later for interpretation and supervised classification. It must stay in original value space apart from minimal cleaning.

Typical operations:

- remove duplicates;
- remove unused identifier columns from model inputs;
- handle required missing-data decisions;
- save a clean reference dataset before clustering-specific transformations.

#### Clustering dataset: `df_prepare_clustering`

This is a dedicated copy used only for clustering transformations.

Processing order:

```text
Clean reference data
        |
        v
Log transformation
        |
        v
StandardScaler
        |
        v
PCA
        |
        v
Clustering
```

Required rules:

- Apply the logarithmic transformation before scaling.
- Never scale first and then apply the log transform.
- Fit clustering models on `df_prepare_clustering`, not on `df_clean`.
- Bring only generated labels back to `df_clean`.

### 3. PCA + Clustering

Two clustering algorithms are studied:

- **K-means**
- **DBSCAN**

#### K-means search

Perform a grid search over:

- several PCA component counts, for example `2..5`;
- several cluster counts `k`, for example `2..7`.

For every combination, compute at least:

- inertia;
- silhouette score.

Then validate the selected configuration with:

- PCA-components × `k` silhouette heatmap;
- elbow curve;
- silhouette curve;
- cluster-size distribution.

A candidate configuration should not contain an excessively small cluster (approximately below 10%) or an overwhelmingly dominant cluster.

#### DBSCAN search

Search across combinations of:

- PCA component count;
- `eps`;
- `min_samples`.

Track:

- silhouette score;
- number of clusters;
- noise proportion;
- Davies-Bouldin score;
- Calinski-Harabasz score where applicable.

DBSCAN noise is represented by label `-1` and should also be exposed through a boolean column such as `est_atypique_dbscan`.

#### Final cluster selection

After comparison, create exactly one final cluster label:

```python
df_clean["cluster_final"] = ...
```

The final choice must consider separation quality, point coverage, and cluster balance. Because the supervised target must exist for every customer, a method that leaves significant noise requires special caution.

### 4. Cluster Interpretation & Business Labels

Cluster numbers such as `0`, `1`, and `2` have no intrinsic business meaning.

For each cluster:

1. Compute mean feature values in standardized space.
2. Compute mean feature values again on original-scale `df_clean` values.
3. Identify the variables that most strongly distinguish the segment.
4. Define interpretable behavioral conditions.
5. Assign a clear business label.
6. Map the cluster number to a final `target` label.
7. Verify that `target` contains no missing values and has the same proportions as the source clusters.

> Cluster IDs may change between runs. Never reuse business names from another run without rebuilding the cluster summaries from the current fitted model.

### 5. Supervised Classification

The classifier starts from **`df_clean + target`**, not from the PCA/log/scaled clustering matrix.

Build:

```text
Raw behavioral features
        |
        v
Train/Test split (stratified)
        |
        v
Scikit-learn Pipeline
  - StandardScaler
  - Classifier
        |
        v
Cross-validation / Hyperparameter search
        |
        v
Evaluation
        |
        v
Serialized complete pipeline (.joblib)
```

Models to compare:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)

Evaluation should include:

- confusion matrix;
- accuracy;
- precision;
- recall;
- F1-score;
- cross-validation stability;
- training time.

Use `GridSearchCV` and/or `RandomizedSearchCV` on the **complete pipeline**, so preprocessing is re-fitted inside every training fold.

If class imbalance is significant, document and test an appropriate `imblearn` strategy.

Save the final object as the full fitted pipeline:

```python
import joblib

joblib.dump(best_pipeline, "models/customer_segment_pipeline.joblib")
```

---

## MLflow Experiment Tracking

Create a dedicated experiment:

```python
mlflow.set_experiment("classification_clients")
```

Each classification training run should log:

- model type;
- tuned hyperparameters;
- accuracy;
- precision;
- recall;
- F1-score;
- fitted scikit-learn pipeline;
- confusion-matrix image;
- classification report artifact.

Launch the tracking UI with:

```bash
uv run mlflow ui
```

Default local interface:

```text
http://127.0.0.1:5000
```

---

## Streamlit Application

The Streamlit application demonstrates the trained classification pipeline with a new customer profile.

It should:

- accept manual customer values and/or a supported input file;
- validate the seven behavioral inputs;
- load the serialized **complete** scikit-learn pipeline;
- call `pipeline.predict(...)` directly on raw input values;
- display the predicted business segment;
- display class probabilities when the selected classifier supports `predict_proba`.

Run locally with:

```bash
uv run streamlit run app/app.py
```

---

## Optional Airflow Orchestration

Airflow is an optional bonus extension. The proposed DAG models the workflow as independent tasks with explicit dependencies:

```text
ingest
  -> preprocess
  -> clustering
  -> label_segments
  -> classification
  -> mlflow_tracking
```

The DAG should define retries, failure handling, scheduling behavior, and observable task states from the Airflow UI.

---

## Expected Visual Deliverables

### EDA

- histograms + KDE with skewness;
- correlation heatmap;
- boxplots per numerical feature;
- missing-value summary.

### Preprocessing / PCA

- post-log histograms;
- before/after skewness table;
- cumulative explained-variance curve with 80% threshold.

### Clustering

- PCA-components × `k` silhouette heatmap;
- elbow curve;
- silhouette curve;
- ranked DBSCAN search table;
- K-means vs DBSCAN comparison table;
- 2D PCA scatter plot colored by `cluster_final`;
- K-means centroids when K-means is retained.

### Cluster interpretation

- standardized cluster-profile heatmap;
- cluster size count and percentage chart;
- real-value comparisons for key behavioral variables.

### Classification

- confusion matrix per tested model;
- model comparison table;
- metric comparison bar chart;
- feature importance or logistic-regression coefficients where relevant;
- cross-validation score distribution.

### MLflow

- experiment run comparison view;
- confusion matrix stored as run artifact.

---

## Recommended Repository Structure

```text
projet-segmentation-clients/
├── data/
│   ├── raw/                       # Original dataset
│   └── processed/                 # df_clean, df_prepare_clustering
├── notebooks/                     # EDA and prototyping
├── src/
│   ├── preprocessing.py
│   ├── clustering.py
│   ├── classification.py
│   └── pipeline.py
├── models/                        # Serialized .joblib pipelines
├── app/
│   └── app.py                     # Streamlit application
├── dags/                          # Optional Airflow DAG
├── mlruns/                        # Local MLflow tracking data
├── tests/                         # Recommended automated checks
├── pyproject.toml                 # uv dependencies and project config
├── uv.lock                        # Locked reproducible versions
├── .gitignore
└── README.md
```

---

## Suggested Module Responsibilities

### `src/preprocessing.py`

- duplicate handling;
- missing-value rules;
- clustering-specific log transform;
- reusable validation helpers;
- creation/persistence of cleaned datasets.

### `src/clustering.py`

- PCA evaluation;
- K-means grid search;
- DBSCAN grid search;
- clustering metrics;
- final label attachment.

### `src/classification.py`

- train/test preparation;
- classifier pipeline creation;
- cross-validation;
- hyperparameter optimization;
- metrics and model comparison;
- model serialization.

### `src/pipeline.py`

- optional high-level orchestration of reusable functions.

### `app/app.py`

- user inputs;
- validation;
- model loading;
- prediction rendering.

---

## Installation

### Prerequisites

- Python 3.11+ recommended
- `uv`
- Git

Clone the project:

```bash
git clone <repository-url>
cd projet-segmentation-clients
```

Create/sync the environment:

```bash
uv sync
```

If the project has not yet been initialized with `uv`:

```bash
uv init
uv add pandas numpy matplotlib seaborn scikit-learn imbalanced-learn mlflow streamlit joblib
```

Optional Airflow dependencies should be added separately only when the bonus orchestration is implemented.

---

## Execution Workflow

A professional implementation order is:

```text
1. Project setup
2. EDA
3. Minimal cleaning -> df_clean
4. Clustering copy -> log -> scaling -> PCA
5. K-means / DBSCAN model search
6. Select cluster_final
7. Interpret clusters and create target
8. Train/test split on df_clean + target
9. Classification pipelines + CV + tuning
10. Save the full best pipeline
11. Track classification experiments with MLflow
12. Build Streamlit application
13. Tests + documentation + reproducibility checks
14. Optional Airflow DAG
```

---

## Quality & Leakage Rules

The following rules are non-negotiable:

1. `df_clean` must remain separate from clustering-specific transformed data.
2. Log transformation, standardization, and PCA for clustering must be fitted on the clustering branch.
3. Classification must not use cluster helper columns as features.
4. Classification preprocessing must be learned on training folds only.
5. Hyperparameter search must wrap the full classification pipeline.
6. The Streamlit app must load the complete serialized pipeline rather than manually duplicating preprocessing logic.
7. Every final customer row must have a valid `target` label before supervised learning.
8. Every non-trivial modeling choice must be documented and reproducible with a fixed `random_state` when the algorithm supports it.

---

## Testing Checklist

Recommended minimum tests:

- dataset contains expected input columns;
- `CUST_ID` uniqueness is checked;
- no accidental duplicate rows remain in `df_clean`;
- transformed clustering matrix contains no invalid infinite values;
- selected PCA matrix has the expected number of components;
- final `cluster_final` contains no missing labels;
- final `target` contains no missing labels;
- classification feature set excludes `target` and all cluster-related columns;
- saved `.joblib` pipeline reloads successfully;
- reloaded pipeline can predict one valid raw customer row;
- Streamlit input schema matches model input schema.

---

## Reproducibility

For deterministic components, define a project-level random seed:

```python
RANDOM_STATE = 42
```

Record:

- package versions in `uv.lock`;
- model hyperparameters;
- clustering search ranges;
- PCA component choice;
- cluster-to-business-label mapping;
- train/test split configuration;
- classification evaluation metrics.

---

## Deliverables

- cleaned dataset (`df_clean`);
- clustering-preparation dataset (`df_prepare_clustering`);
- EDA and modeling visualizations;
- K-means and DBSCAN comparison results;
- final `cluster_final` and business `target` labels;
- classification comparison report;
- best serialized scikit-learn pipeline (`.joblib`);
- MLflow experiment runs and artifacts;
- Streamlit application;
- project documentation;
- Jira project plan;
- optional Airflow DAG.

---

## Definition of Done

The project is considered complete when:

- all mandatory feature stories are implemented;
- clustering and classification preprocessing are correctly separated;
- the final segmentation is interpretable and documented;
- every customer has a valid final business target;
- multiple classifiers are evaluated with cross-validation;
- the best complete pipeline is serialized and reloadable;
- MLflow contains reproducible classification runs;
- Streamlit predicts a segment from raw customer inputs;
- a third party can reproduce the project from the repository documentation and locked dependencies.

---

## License

Add the appropriate project license if required by the organization or training program.
