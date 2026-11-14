from pathlib import Path
import json
import warnings

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.decomposition import TruncatedSVD


warnings.filterwarnings("ignore")


# ============================================================
# 1. PROJECT PATHS AND SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
NUMBER_OF_FACTORS = 50
TOP_K = 5


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

def load_training_data():

    print("Loading training data...")

    train = pd.read_csv(
        PROCESSED_DIR / "train_interactions.csv"
    )

    items = pd.read_csv(
        PROCESSED_DIR / "valid_items.csv"
    )

    print(f"Training interactions : {len(train)}")
    print(f"Available movies      : {len(items)}")
    print(f"Unique users          : {train['user_id'].nunique()}")
    print(f"Rated movies          : {train['movie_id'].nunique()}")

    return train, items


# ============================================================
# 3. CONFIGURE MLFLOW
# ============================================================

def configure_mlflow():

    database_path = PROJECT_ROOT / "mlflow.db"

    tracking_uri = (
        f"sqlite:///{database_path.as_posix()}"
    )

    mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(
        "movie_recommendation_experiments"
    )

    print("\nMLflow configured successfully.")
    print(f"Tracking URI: {tracking_uri}")


# ============================================================
# 4. TRAIN POPULARITY MODEL
# ============================================================

def train_popularity_model(train, items):

    print("\n" + "-" * 60)
    print("TRAINING MODEL 1: POPULARITY RECOMMENDER")
    print("-" * 60)

    popularity = (
        train
        .groupby("movie_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    popularity["average_rating"] = (
        popularity["average_rating"].round(4)
    )

    # Weighted popularity score
    popularity["popularity_score"] = (
        popularity["average_rating"]
        * np.log1p(popularity["rating_count"])
    )

    popularity = popularity.merge(
        items[["movie_id", "movie_title"]],
        on="movie_id",
        how="left"
    )

    popularity = popularity.sort_values(
        by="popularity_score",
        ascending=False
    ).reset_index(drop=True)

    model_file = MODEL_DIR / "popularity_model.csv"

    popularity.to_csv(
        model_file,
        index=False
    )

    with mlflow.start_run(
        run_name="popularity_recommender"
    ):

        mlflow.log_param(
            "model_type",
            "popularity_baseline"
        )

        mlflow.log_param(
            "top_k",
            TOP_K
        )

        mlflow.log_param(
            "score_formula",
            "average_rating * log(1 + rating_count)"
        )

        mlflow.log_metric(
            "training_interactions",
            len(train)
        )

        mlflow.log_metric(
            "movies_ranked",
            len(popularity)
        )

        mlflow.log_artifact(
            str(model_file)
        )

    print("Popularity model trained successfully.")
    print(f"Movies ranked: {len(popularity)}")

    print("\nTop 5 popularity recommendations:")

    print(
        popularity[
            [
                "movie_id",
                "movie_title",
                "rating_count",
                "average_rating",
                "popularity_score"
            ]
        ]
        .head(TOP_K)
        .to_string(index=False)
    )

    return popularity


# ============================================================
# 5. CREATE USER-ITEM MATRIX
# ============================================================

def create_user_item_matrix(train):

    print("\nCreating training user-item matrix...")

    matrix = train.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )

    user_ids = matrix.index.to_numpy()
    movie_ids = matrix.columns.to_numpy()

    matrix_values = matrix.to_numpy(
        dtype=np.float64
    )

    print(f"Matrix shape: {matrix_values.shape}")

    return (
        matrix_values,
        user_ids,
        movie_ids
    )


# ============================================================
# 6. TRAIN PERSONALIZED SVD MODEL
# ============================================================

def train_svd_model(
    matrix_values,
    user_ids,
    movie_ids
):

    print("\n" + "-" * 60)
    print("TRAINING MODEL 2: PERSONALIZED SVD RECOMMENDER")
    print("-" * 60)

    maximum_factors = min(
        matrix_values.shape[0] - 1,
        matrix_values.shape[1] - 1
    )

    factors = min(
        NUMBER_OF_FACTORS,
        maximum_factors
    )

    svd = TruncatedSVD(
        n_components=factors,
        random_state=RANDOM_STATE
    )

    with mlflow.start_run(
        run_name="personalized_svd_recommender"
    ):

        mlflow.log_param(
            "model_type",
            "truncated_svd"
        )

        mlflow.log_param(
            "number_of_factors",
            factors
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE
        )

        mlflow.log_param(
            "top_k",
            TOP_K
        )

        # Learn compressed user representation
        user_factors = svd.fit_transform(
            matrix_values
        )

        # Reconstruct estimated user-movie ratings
        predicted_ratings = np.dot(
            user_factors,
            svd.components_
        )

        explained_variance = (
            svd.explained_variance_ratio_.sum()
        )

        mlflow.log_metric(
            "explained_variance_ratio",
            float(explained_variance)
        )

        mlflow.log_metric(
            "number_of_users",
            len(user_ids)
        )

        mlflow.log_metric(
            "number_of_movies",
            len(movie_ids)
        )

        # Save complete model package
        model_package = {
            "svd_model": svd,
            "user_factors": user_factors,
            "predicted_ratings": predicted_ratings,
            "original_matrix": matrix_values,
            "user_ids": user_ids,
            "movie_ids": movie_ids
        }

        model_file = MODEL_DIR / "svd_model.joblib"

        joblib.dump(
            model_package,
            model_file
        )

        mlflow.log_artifact(
            str(model_file)
        )

    print("Personalized SVD model trained successfully.")

    print(
        "Explained variance ratio: "
        f"{explained_variance:.4f}"
    )

    print(f"Latent factors used: {factors}")

    return model_package


# ============================================================
# 7. GENERATE SAMPLE PERSONALIZED RECOMMENDATIONS
# ============================================================

def generate_sample_recommendations(
    model_package,
    items,
    requested_user_id=1
):

    print("\nGenerating sample personalized recommendations...")

    user_ids = model_package["user_ids"]
    movie_ids = model_package["movie_ids"]
    predicted_ratings = model_package[
        "predicted_ratings"
    ]
    original_matrix = model_package[
        "original_matrix"
    ]

    if requested_user_id not in user_ids:
        requested_user_id = int(user_ids[0])

    user_position = np.where(
        user_ids == requested_user_id
    )[0][0]

    user_predictions = predicted_ratings[
        user_position
    ].copy()

    already_rated_mask = (
        original_matrix[user_position] > 0
    )

    # Prevent movies already rated by the user
    user_predictions[already_rated_mask] = -np.inf

    top_positions = np.argsort(
        user_predictions
    )[::-1][:TOP_K]

    recommended_movie_ids = movie_ids[
        top_positions
    ]

    recommendation_scores = user_predictions[
        top_positions
    ]

    recommendations = pd.DataFrame(
        {
            "user_id": requested_user_id,
            "movie_id": recommended_movie_ids,
            "predicted_score": recommendation_scores
        }
    )

    recommendations = recommendations.merge(
        items[["movie_id", "movie_title"]],
        on="movie_id",
        how="left"
    )

    recommendations = recommendations[
        [
            "user_id",
            "movie_id",
            "movie_title",
            "predicted_score"
        ]
    ]

    recommendations["predicted_score"] = (
        recommendations["predicted_score"]
        .round(4)
    )

    output_file = (
        REPORT_DIR
        / "sample_personalized_recommendations.csv"
    )

    recommendations.to_csv(
        output_file,
        index=False
    )

    print(
        f"\nTop {TOP_K} recommendations "
        f"for User {requested_user_id}:"
    )

    print(
        recommendations.to_string(index=False)
    )

    return recommendations


# ============================================================
# 8. SAVE TRAINING SUMMARY
# ============================================================

def save_training_summary(
    popularity,
    model_package
):

    summary = {
        "popularity_model": {
            "model_type": "Popularity baseline",
            "movies_ranked": int(len(popularity)),
            "top_k": TOP_K
        },
        "personalized_model": {
            "model_type": "Truncated SVD",
            "number_of_users": int(
                len(model_package["user_ids"])
            ),
            "number_of_movies": int(
                len(model_package["movie_ids"])
            ),
            "number_of_factors": int(
                model_package[
                    "svd_model"
                ].n_components
            ),
            "explained_variance_ratio": float(
                model_package[
                    "svd_model"
                ].explained_variance_ratio_.sum()
            ),
            "top_k": TOP_K
        }
    }

    summary_file = (
        REPORT_DIR / "training_summary.json"
    )

    with open(
        summary_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    print(
        f"\nTraining summary saved: {summary_file}"
    )


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("MODEL TRAINING STARTED")
    print("=" * 60)

    train, items = load_training_data()

    configure_mlflow()

    popularity = train_popularity_model(
        train,
        items
    )

    (
        matrix_values,
        user_ids,
        movie_ids
    ) = create_user_item_matrix(train)

    model_package = train_svd_model(
        matrix_values,
        user_ids,
        movie_ids
    )

    generate_sample_recommendations(
        model_package,
        items,
        requested_user_id=1
    )

    save_training_summary(
        popularity,
        model_package
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nTRAINED MODELS")

    print(
        "1. Popularity-based recommender"
    )

    print(
        "2. Personalized Truncated SVD recommender"
    )


if __name__ == "__main__":
    main()