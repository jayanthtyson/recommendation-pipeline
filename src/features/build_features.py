from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURE_DIR = PROJECT_ROOT / "data" / "features"
REPORT_DIR = PROJECT_ROOT / "reports"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD PREPARED DATA
# ============================================================

def load_data():

    print("Loading prepared datasets...")

    users = pd.read_csv(
        PROCESSED_DIR / "valid_users.csv"
    )

    items = pd.read_csv(
        PROCESSED_DIR / "valid_items.csv"
    )

    train = pd.read_csv(
        PROCESSED_DIR / "train_interactions.csv"
    )

    print(f"Users                 : {len(users)}")
    print(f"Movies                : {len(items)}")
    print(f"Training interactions : {len(train)}")

    return users, items, train


# ============================================================
# 3. CREATE USER FEATURES
# ============================================================

def create_user_features(users, train):

    print("\nCreating user features...")

    activity = (
        train
        .groupby("user_id")
        .agg(
            user_activity_count=("movie_id", "count"),
            user_average_rating=("rating", "mean"),
            last_interaction=("interaction_date", "max")
        )
        .reset_index()
    )

    activity["user_average_rating"] = (
        activity["user_average_rating"].round(2)
    )

    user_features = users.merge(
        activity,
        on="user_id",
        how="left"
    )

    print(
        f"User feature records created: "
        f"{len(user_features)}"
    )

    return user_features


# ============================================================
# 4. CREATE ITEM FEATURES
# ============================================================

def create_item_features(items, train):

    print("\nCreating item features...")

    popularity = (
        train
        .groupby("movie_id")
        .agg(
            item_popularity=("user_id", "count"),
            item_average_rating=("rating", "mean")
        )
        .reset_index()
    )

    popularity["item_average_rating"] = (
        popularity["item_average_rating"].round(2)
    )

    item_features = items.merge(
        popularity,
        on="movie_id",
        how="left"
    )

    item_features["item_popularity"] = (
        item_features["item_popularity"]
        .fillna(0)
    )

    print(
        f"Item feature records created: "
        f"{len(item_features)}"
    )

    return item_features


# ============================================================
# 5. CREATE INTERACTION FEATURES
# ============================================================

def create_interaction_features(train):

    print("\nCreating interaction features...")

    interaction_features = train.copy()

    interaction_features["interaction_date"] = (
        pd.to_datetime(
            interaction_features["interaction_date"]
        )
    )

    # User's average training rating
    user_average = (
        interaction_features
        .groupby("user_id")["rating"]
        .transform("mean")
    )

    interaction_features["rating_deviation"] = (
        interaction_features["rating"]
        - user_average
    ).round(2)

    # Number of days since each user's previous interaction
    interaction_features = (
        interaction_features
        .sort_values(
            ["user_id", "interaction_date"]
        )
    )

    interaction_features[
        "days_since_last_interaction"
    ] = (
        interaction_features
        .groupby("user_id")["interaction_date"]
        .diff()
        .dt.total_seconds()
        .div(86400)
        .fillna(0)
        .round(2)
    )

    print(
        "Interaction feature records created: "
        f"{len(interaction_features)}"
    )

    return interaction_features


# ============================================================
# 6. CREATE FEATURE CATALOG
# ============================================================

def create_feature_catalog():

    print("\nCreating feature catalog...")

    catalog = pd.DataFrame(
        [
            {
                "entity": "User",
                "feature": "user_activity_count",
                "description":
                    "Number of movie ratings made by the user",
                "source":
                    "Training interactions"
            },
            {
                "entity": "User",
                "feature": "user_average_rating",
                "description":
                    "Average rating given by the user",
                "source":
                    "Training interactions"
            },
            {
                "entity": "User",
                "feature": "last_interaction",
                "description":
                    "Most recent training interaction for the user",
                "source":
                    "Training interactions"
            },
            {
                "entity": "Item",
                "feature": "item_popularity",
                "description":
                    "Number of training ratings received by the movie",
                "source":
                    "Training interactions"
            },
            {
                "entity": "Item",
                "feature": "item_average_rating",
                "description":
                    "Average training rating received by the movie",
                "source":
                    "Training interactions"
            },
            {
                "entity": "Interaction",
                "feature": "rating_deviation",
                "description":
                    "Rating minus the user's average rating",
                "source":
                    "Training interactions"
            },
            {
                "entity": "Interaction",
                "feature": "days_since_last_interaction",
                "description":
                    "Days since the user's previous interaction",
                "source":
                    "Training interactions"
            }
        ]
    )

    return catalog


# ============================================================
# 7. SAVE RESULTS
# ============================================================

def save_features(
    user_features,
    item_features,
    interaction_features,
    catalog
):

    print("\nSaving feature datasets...")

    user_features.to_csv(
        FEATURE_DIR / "user_features.csv",
        index=False
    )

    item_features.to_csv(
        FEATURE_DIR / "item_features.csv",
        index=False
    )

    interaction_features.to_csv(
        FEATURE_DIR / "interaction_features.csv",
        index=False
    )

    catalog.to_csv(
        REPORT_DIR / "feature_catalog.csv",
        index=False
    )

    print("Feature datasets saved successfully.")


# ============================================================
# 8. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("FEATURE MANAGEMENT STARTED")
    print("=" * 60)

    users, items, train = load_data()

    user_features = create_user_features(
        users,
        train
    )

    item_features = create_item_features(
        items,
        train
    )

    interaction_features = (
        create_interaction_features(train)
    )

    catalog = create_feature_catalog()

    save_features(
        user_features,
        item_features,
        interaction_features,
        catalog
    )

    print("\n" + "=" * 60)
    print("FEATURE MANAGEMENT COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nFINAL FEATURE SUMMARY")

    print(
        f"User features        : {len(user_features)}"
    )

    print(
        f"Item features        : {len(item_features)}"
    )

    print(
        "Interaction features : "
        f"{len(interaction_features)}"
    )

    print(
        f"Feature catalog      : {len(catalog)} features"
    )


if __name__ == "__main__":
    main()