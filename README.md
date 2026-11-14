# End-to-End Data Management Pipeline for a Recommendation System

## Overview

This project implements an end-to-end data management and machine
learning pipeline for a movie recommendation system using the
MovieLens 100K dataset.

The pipeline covers:

1. Data ingestion
2. Data validation
3. Data preparation and transformation
4. Feature management
5. Model training
6. Experiment tracking
7. Model evaluation
8. Workflow orchestration

## Dataset

The project uses the MovieLens 100K dataset.

The dataset contains:

- 943 users
- 1,682 movies
- 100,000 user-movie rating interactions
- Ratings from 1 to 5

The main source files are:

- `u.user` - user information
- `u.item` - movie information
- `u.data` - user-movie ratings

## Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- MLflow
- Joblib
- Prefect
- CSV
- SQLite

## Project Structure

recommendation-pipeline/

- `data/raw` - original MovieLens data
- `data/processed` - validated and prepared datasets
- `data/errors` - invalid records
- `data/features` - generated feature datasets
- `src/ingestion` - data ingestion
- `src/validation` - automated data-quality validation
- `src/preparation` - data preparation and train/test splitting
- `src/features` - feature generation
- `src/training` - recommendation model training
- `src/evaluation` - model evaluation
- `orchestration` - Prefect workflow
- `models` - trained model artifacts
- `reports` - evaluation results and reports
- `screenshots` - pipeline execution evidence

## Data Ingestion

The ingestion stage reads MovieLens user, movie, and rating files
and converts them into standardized CSV datasets.

Ingested records:

- Users: 943
- Movies: 1,682
- Interactions: 100,000

Run:

python src/ingestion/ingest_data.py

## Data Validation

The validation stage performs automated checks for:

- Required columns
- Missing values
- Duplicate records
- Invalid user ages
- Ratings outside the 1-5 range
- Invalid timestamps
- Unknown users
- Unknown movies
- Referential integrity

Invalid records are written to `data/errors`.

Validated records are written to `data/processed`.

## Data Preparation

User interactions are sorted chronologically and split using a
per-user time-based 80/20 train/test approach.

Execution produced:

- Training interactions: 79,619
- Testing interactions: 20,381
- Users: 943
- Movies represented in training: 1,613

Derived datasets include:

- User activity summary
- Item popularity summary
- User-item interaction matrix

## Feature Management

Reusable user, item, and interaction features are generated.

Examples include:

- User activity count
- User average rating
- Last interaction
- Item popularity
- Item average rating
- Rating deviation
- Days since previous interaction

The feature catalog is available at:

reports/feature_catalog.csv

## Recommendation Models

Two recommendation models were implemented.

### Popularity-Based Recommender

The baseline model ranks movies using:

average_rating * log(1 + rating_count)

### Personalized Recommender

A personalized recommendation model was implemented using
Scikit-learn TruncatedSVD.

Configuration:

- Latent factors: 50
- Random state: 42
- Recommendation size: Top 5

## MLflow Experiment Tracking

MLflow is used to track model parameters, metrics, and artifacts.

The experiment is named:

movie_recommendation_experiments

The experiment includes runs for:

- popularity_recommender
- personalized_svd_recommender

MLflow metadata is stored using a local SQLite database.

## Model Evaluation

Both recommenders were evaluated using ranking metrics.

| Metric | Popularity | Personalized SVD |
|---|---:|---:|
| Precision@5 | 0.0975 | 0.1488 |
| Recall@5 | 0.0434 | 0.0966 |
| NDCG@5 | 0.1058 | 0.1696 |
| Coverage | 0.0285 | 0.2474 |
| Diversity | 0.4418 | 0.7212 |
| RMSE | N/A | 2.4756 |
| MAE | N/A | 2.1941 |

The personalized SVD recommender performed better than the
popularity baseline across the evaluated ranking metrics.

## Workflow Orchestration

Prefect orchestrates the complete workflow:

Data Ingestion
→ Data Validation
→ Data Preparation
→ Feature Management
→ Model Training
→ Model Evaluation

Run the complete workflow using:

python orchestration/pipeline_flow.py

The successful end-to-end execution processed all pipeline stages
in approximately 25 seconds during the recorded assignment run.

## Installation

Create a Python virtual environment.

Windows:

python -m venv .venv

Activate it:

.venv\Scripts\activate

Install dependencies:

pip install -r requirements.txt

## Run Individual Pipeline Stages

python src/ingestion/ingest_data.py

python src/validation/validate_data.py

python src/preparation/prepare_data.py

python src/features/build_features.py

python src/training/train_model.py

python src/evaluation/evaluate_models.py

## Run Complete Pipeline

python orchestration/pipeline_flow.py

## MLflow UI

Start the local MLflow server:

python -m mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 8080

Then open:

http://127.0.0.1:8080

## Key Outputs

- `reports/logs/validation_report.txt`
- `reports/feature_catalog.csv`
- `reports/model_comparison.csv`
- `reports/training_summary.json`
- `models/popularity_model.csv`
- `models/svd_model.joblib`

## Limitations

The project uses the MovieLens 100K dataset, which is relatively
small and historical.

The SVD implementation uses a sparse user-item rating matrix with
zero values for missing interactions. More advanced matrix
factorization approaches could improve rating prediction.

The diversity metric is implemented as an inverse-popularity
measure rather than feature-based or genre-distance diversity.

The current implementation is designed as an educational,
reproducible local data pipeline rather than a production-scale
recommendation service.

## Conclusion

The project demonstrates a reproducible end-to-end data management
pipeline for recommendation systems.

The pipeline successfully integrates ingestion, validation,
preparation, feature management, recommendation model training,
experiment tracking, model evaluation, and Prefect orchestration.