from pathlib import Path
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD VALIDATED DATA
# ============================================================

def load_validated_data():

    print("Loading validated datasets...")

    users = pd.read_csv(
        PROCESSED_DIR / "valid_users.csv"
    )

    items = pd.read_csv(
        PROCESSED_DIR / "valid_items.csv"
    )

    interactions = pd.read_csv(
        PROCESSED_DIR / "valid_interactions.csv"
    )

    print(f"Users loaded        : {len(users)}")
    print(f"Movies loaded       : {len(items)}")
    print(f"Interactions loaded : {len(interactions)}")

    return users, items, interactions


# ============================================================
# 3. PREPARE INTERACTIONS
# ============================================================

def prepare_interactions(interactions):

    print("\nPreparing interaction data...")

    prepared = interactions.copy()

    # Convert Unix timestamp into readable date/time
    prepared["interaction_date"] = pd.to_datetime(
        prepared["timestamp"],
        unit="s"
    )

    # Sort by user and time
    prepared = prepared.sort_values(
        by=["user_id", "timestamp"]
    ).reset_index(drop=True)

    print("Timestamp converted successfully.")
    print("Interactions sorted chronologically.")

    return prepared


# ============================================================
# 4. TIME-BASED TRAIN/TEST SPLIT
# ============================================================

def time_based_split(interactions, train_ratio=0.80):

    print("\nCreating time-based train/test split...")

    train_parts = []
    test_parts = []

    # Process each user separately
    for user_id, user_data in interactions.groupby("user_id"):

        user_data = user_data.sort_values(
            "timestamp"
        ).reset_index(drop=True)

        total_interactions = len(user_data)

        # Calculate split point
        split_index = int(
            total_interactions * train_ratio
        )

        # Make sure both train and test contain data
        if split_index <= 0:
            split_index = 1

        if split_index >= total_interactions:
            split_index = total_interactions - 1

        train_user = user_data.iloc[
            :split_index
        ].copy()

        test_user = user_data.iloc[
            split_index:
        ].copy()

        train_parts.append(train_user)
        test_parts.append(test_user)

    train_data = pd.concat(
        train_parts,
        ignore_index=True
    )

    test_data = pd.concat(
        test_parts,
        ignore_index=True
    )

    print(
        f"Training interactions : {len(train_data)}"
    )

    print(
        f"Testing interactions  : {len(test_data)}"
    )

    return train_data, test_data


# ============================================================
# 5. CREATE USER ACTIVITY SUMMARY
# ============================================================

def create_user_activity_summary(train_data):

    print("\nCreating user activity summary...")

    user_summary = (
        train_data
        .groupby("user_id")
        .agg(
            activity_count=("movie_id", "count"),
            average_rating=("rating", "mean"),
            last_interaction=(
                "interaction_date",
                "max"
            )
        )
        .reset_index()
    )

    user_summary["average_rating"] = (
        user_summary["average_rating"]
        .round(2)
    )

    print(
        f"User summaries created: {len(user_summary)}"
    )

    return user_summary


# ============================================================
# 6. CREATE ITEM POPULARITY SUMMARY
# ============================================================

def create_item_popularity(train_data, items):

    print("\nCreating item popularity summary...")

    item_popularity = (
        train_data
        .groupby("movie_id")
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean")
        )
        .reset_index()
    )

    item_popularity["average_rating"] = (
        item_popularity["average_rating"]
        .round(2)
    )

    # Add movie title
    item_popularity = item_popularity.merge(
        items[["movie_id", "movie_title"]],
        on="movie_id",
        how="left"
    )

    # Most popular movies first
    item_popularity = item_popularity.sort_values(
        by="rating_count",
        ascending=False
    )

    print(
        "Item popularity summary created: "
        f"{len(item_popularity)} movies"
    )

    return item_popularity


# ============================================================
# 7. CREATE USER-ITEM MATRIX
# ============================================================

def create_user_item_matrix(train_data):

    print("\nCreating user-item interaction matrix...")

    user_item_matrix = train_data.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating",
        aggfunc="mean",
        fill_value=0
    )

    print(
        "User-item matrix shape: "
        f"{user_item_matrix.shape}"
    )

    return user_item_matrix


# ============================================================
# 8. SAVE ALL PREPARED DATA
# ============================================================

def save_outputs(
    prepared_interactions,
    train_data,
    test_data,
    user_summary,
    item_popularity,
    user_item_matrix
):

    print("\nSaving preparation outputs...")

    prepared_interactions.to_csv(
        PROCESSED_DIR / "prepared_interactions.csv",
        index=False
    )

    train_data.to_csv(
        PROCESSED_DIR / "train_interactions.csv",
        index=False
    )

    test_data.to_csv(
        PROCESSED_DIR / "test_interactions.csv",
        index=False
    )

    user_summary.to_csv(
        PROCESSED_DIR / "user_activity_summary.csv",
        index=False
    )

    item_popularity.to_csv(
        PROCESSED_DIR / "item_popularity.csv",
        index=False
    )

    user_item_matrix.to_csv(
        PROCESSED_DIR / "user_item_matrix.csv"
    )

    print("All preparation outputs saved successfully.")


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("DATA PREPARATION STARTED")
    print("=" * 60)

    users, items, interactions = (
        load_validated_data()
    )

    prepared_interactions = (
        prepare_interactions(interactions)
    )

    train_data, test_data = time_based_split(
        prepared_interactions,
        train_ratio=0.80
    )

    user_summary = (
        create_user_activity_summary(train_data)
    )

    item_popularity = (
        create_item_popularity(
            train_data,
            items
        )
    )

    user_item_matrix = (
        create_user_item_matrix(train_data)
    )

    save_outputs(
        prepared_interactions,
        train_data,
        test_data,
        user_summary,
        item_popularity,
        user_item_matrix
    )

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nFINAL PREPARATION SUMMARY")

    print(
        f"Total interactions : "
        f"{len(prepared_interactions)}"
    )

    print(
        f"Training records    : "
        f"{len(train_data)}"
    )

    print(
        f"Testing records     : "
        f"{len(test_data)}"
    )

    print(
        f"Users summarized    : "
        f"{len(user_summary)}"
    )

    print(
        f"Movies summarized   : "
        f"{len(item_popularity)}"
    )

    print(
        "User-item matrix    : "
        f"{user_item_matrix.shape}"
    )


if __name__ == "__main__":
    main()