from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# 1. PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

TOP_K = 5
RELEVANT_RATING = 4


# ============================================================
# 2. LOAD DATA AND MODELS
# ============================================================

def load_resources():

    print("Loading evaluation resources...")

    train = pd.read_csv(
        PROCESSED_DIR / "train_interactions.csv"
    )

    test = pd.read_csv(
        PROCESSED_DIR / "test_interactions.csv"
    )

    popularity = pd.read_csv(
        MODEL_DIR / "popularity_model.csv"
    )

    svd_package = joblib.load(
        MODEL_DIR / "svd_model.joblib"
    )

    print(f"Training interactions : {len(train)}")
    print(f"Testing interactions  : {len(test)}")
    print(f"Popularity items      : {len(popularity)}")

    return train, test, popularity, svd_package


# ============================================================
# 3. PRECISION / RECALL / NDCG
# ============================================================

def ranking_metrics(recommended, relevant, k):

    recommended = recommended[:k]

    relevant = set(relevant)

    if len(relevant) == 0:
        return None, None, None

    hits = [
        1 if movie_id in relevant else 0
        for movie_id in recommended
    ]

    precision = sum(hits) / k

    recall = sum(hits) / len(relevant)

    dcg = 0.0

    for rank, hit in enumerate(hits, start=1):

        if hit:
            dcg += 1 / math.log2(rank + 1)

    ideal_hits = min(
        len(relevant),
        k
    )

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_hits + 1)
    )

    ndcg = dcg / idcg if idcg > 0 else 0.0

    return precision, recall, ndcg


# ============================================================
# 4. POPULARITY RECOMMENDATIONS
# ============================================================

def get_popularity_recommendations(
    user_id,
    train,
    popularity,
    k
):

    already_seen = set(
        train.loc[
            train["user_id"] == user_id,
            "movie_id"
        ]
    )

    candidates = popularity[
        ~popularity["movie_id"].isin(
            already_seen
        )
    ]

    return (
        candidates["movie_id"]
        .head(k)
        .tolist()
    )


# ============================================================
# 5. SVD RECOMMENDATIONS
# ============================================================

def get_svd_recommendations(
    user_id,
    svd_package,
    k
):

    user_ids = svd_package["user_ids"]
    movie_ids = svd_package["movie_ids"]

    predictions = svd_package[
        "predicted_ratings"
    ]

    original_matrix = svd_package[
        "original_matrix"
    ]

    positions = np.where(
        user_ids == user_id
    )[0]

    if len(positions) == 0:
        return []

    user_position = positions[0]

    user_scores = predictions[
        user_position
    ].copy()

    already_seen = (
        original_matrix[user_position] > 0
    )

    user_scores[already_seen] = -np.inf

    top_positions = np.argsort(
        user_scores
    )[::-1][:k]

    return movie_ids[
        top_positions
    ].tolist()


# ============================================================
# 6. EVALUATE RANKING MODEL
# ============================================================

def evaluate_ranking_model(
    model_name,
    recommendation_function,
    train,
    test,
    k
):

    print(
        f"\nEvaluating {model_name}..."
    )

    precisions = []
    recalls = []
    ndcgs = []

    all_recommended_items = set()

    evaluated_users = 0

    for user_id, user_test in test.groupby(
        "user_id"
    ):

        relevant_items = user_test.loc[
            user_test["rating"] >= RELEVANT_RATING,
            "movie_id"
        ].tolist()

        if len(relevant_items) == 0:
            continue

        recommendations = (
            recommendation_function(
                user_id
            )
        )

        if len(recommendations) == 0:
            continue

        precision, recall, ndcg = (
            ranking_metrics(
                recommendations,
                relevant_items,
                k
            )
        )

        precisions.append(precision)
        recalls.append(recall)
        ndcgs.append(ndcg)

        all_recommended_items.update(
            recommendations
        )

        evaluated_users += 1

    total_catalog_items = (
        train["movie_id"].nunique()
    )

    coverage = (
        len(all_recommended_items)
        / total_catalog_items
    )

    results = {
        "model": model_name,
        "precision_at_5": np.mean(precisions),
        "recall_at_5": np.mean(recalls),
        "ndcg_at_5": np.mean(ndcgs),
        "coverage": coverage,
        "evaluated_users": evaluated_users
    }

    return results, all_recommended_items


# ============================================================
# 7. SIMPLE DIVERSITY MEASURE
# ============================================================

def calculate_diversity(
    recommended_items,
    train
):

    """
    Diversity is estimated using inverse popularity.

    Rarely rated movies receive higher diversity scores.
    """

    popularity_counts = (
        train["movie_id"]
        .value_counts()
    )

    maximum_popularity = (
        popularity_counts.max()
    )

    diversity_scores = []

    for movie_id in recommended_items:

        item_popularity = (
            popularity_counts.get(
                movie_id,
                0
            )
        )

        diversity_score = (
            1
            - (
                item_popularity
                / maximum_popularity
            )
        )

        diversity_scores.append(
            diversity_score
        )

    if len(diversity_scores) == 0:
        return 0.0

    return float(
        np.mean(diversity_scores)
    )


# ============================================================
# 8. RMSE AND MAE FOR SVD
# ============================================================

def calculate_svd_rating_metrics(
    test,
    svd_package
):

    print("\nCalculating SVD RMSE and MAE...")

    user_ids = svd_package["user_ids"]
    movie_ids = svd_package["movie_ids"]

    predictions = svd_package[
        "predicted_ratings"
    ]

    user_lookup = {
        user_id: index
        for index, user_id
        in enumerate(user_ids)
    }

    movie_lookup = {
        movie_id: index
        for index, movie_id
        in enumerate(movie_ids)
    }

    actual_values = []
    predicted_values = []

    for row in test.itertuples():

        if (
            row.user_id not in user_lookup
            or row.movie_id not in movie_lookup
        ):
            continue

        user_position = user_lookup[
            row.user_id
        ]

        movie_position = movie_lookup[
            row.movie_id
        ]

        predicted_rating = predictions[
            user_position,
            movie_position
        ]

        # Keep predictions within rating scale
        predicted_rating = np.clip(
            predicted_rating,
            1,
            5
        )

        actual_values.append(
            row.rating
        )

        predicted_values.append(
            predicted_rating
        )

    rmse = np.sqrt(
        mean_squared_error(
            actual_values,
            predicted_values
        )
    )

    mae = mean_absolute_error(
        actual_values,
        predicted_values
    )

    return float(rmse), float(mae)


# ============================================================
# 9. MAIN EVALUATION
# ============================================================

def main():

    print("=" * 60)
    print("MODEL EVALUATION STARTED")
    print("=" * 60)

    (
        train,
        test,
        popularity,
        svd_package
    ) = load_resources()

    popularity_function = lambda user_id: (
        get_popularity_recommendations(
            user_id,
            train,
            popularity,
            TOP_K
        )
    )

    svd_function = lambda user_id: (
        get_svd_recommendations(
            user_id,
            svd_package,
            TOP_K
        )
    )

    popularity_results, popularity_items = (
        evaluate_ranking_model(
            "Popularity",
            popularity_function,
            train,
            test,
            TOP_K
        )
    )

    svd_results, svd_items = (
        evaluate_ranking_model(
            "Personalized SVD",
            svd_function,
            train,
            test,
            TOP_K
        )
    )

    popularity_results["diversity"] = (
        calculate_diversity(
            popularity_items,
            train
        )
    )

    svd_results["diversity"] = (
        calculate_diversity(
            svd_items,
            train
        )
    )

    rmse, mae = calculate_svd_rating_metrics(
        test,
        svd_package
    )

    popularity_results["rmse"] = np.nan
    popularity_results["mae"] = np.nan

    svd_results["rmse"] = rmse
    svd_results["mae"] = mae

    results = pd.DataFrame(
        [
            popularity_results,
            svd_results
        ]
    )

    metric_columns = [
        "precision_at_5",
        "recall_at_5",
        "ndcg_at_5",
        "coverage",
        "diversity",
        "rmse",
        "mae"
    ]

    results[metric_columns] = (
        results[metric_columns].round(4)
    )

    output_file = (
        REPORT_DIR
        / "model_comparison.csv"
    )

    results.to_csv(
        output_file,
        index=False
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results.to_string(
            index=False
        )
    )

    print("\n" + "=" * 60)
    print(
        "MODEL EVALUATION COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)

    print(
        f"\nResults saved to: {output_file}"
    )


if __name__ == "__main__":
    main()