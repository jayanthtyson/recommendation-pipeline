from pathlib import Path
import pandas as pd


# ---------------------------------------------------
# 1. Find project folders
# ---------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"


# ---------------------------------------------------
# 2. Ingest Users
# ---------------------------------------------------

def ingest_users():

    print("Reading user data...")

    user_columns = [
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code"
    ]

    users = pd.read_csv(
        RAW_DIR / "u.user",
        sep="|",
        names=user_columns,
        encoding="latin-1"
    )

    output_file = RAW_DIR / "users.csv"

    users.to_csv(
        output_file,
        index=False
    )

    print(f"Users loaded: {len(users)}")
    print("Created users.csv")

    return users


# ---------------------------------------------------
# 3. Ingest Movies
# ---------------------------------------------------

def ingest_items():

    print("\nReading movie data...")

    item_columns = [
        "movie_id",
        "movie_title",
        "release_date",
        "video_release_date",
        "imdb_url",
        "unknown",
        "action",
        "adventure",
        "animation",
        "childrens",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film_noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci_fi",
        "thriller",
        "war",
        "western"
    ]

    items = pd.read_csv(
        RAW_DIR / "u.item",
        sep="|",
        names=item_columns,
        encoding="latin-1"
    )

    output_file = RAW_DIR / "items.csv"

    items.to_csv(
        output_file,
        index=False
    )

    print(f"Movies loaded: {len(items)}")
    print("Created items.csv")

    return items


# ---------------------------------------------------
# 4. Ingest User-Movie Interactions
# ---------------------------------------------------

def ingest_interactions():

    print("\nReading interaction data...")

    interaction_columns = [
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    ]

    interactions = pd.read_csv(
        RAW_DIR / "u.data",
        sep="\t",
        names=interaction_columns
    )

    output_file = RAW_DIR / "interactions.csv"

    interactions.to_csv(
        output_file,
        index=False
    )

    print(
        f"Interactions loaded: {len(interactions)}"
    )

    print("Created interactions.csv")

    return interactions


# ---------------------------------------------------
# 5. Main ingestion pipeline
# ---------------------------------------------------

def main():

    print("=" * 50)
    print("DATA INGESTION STARTED")
    print("=" * 50)

    users = ingest_users()

    items = ingest_items()

    interactions = ingest_interactions()

    print("\n" + "=" * 50)

    print(
        "DATA INGESTION COMPLETED SUCCESSFULLY"
    )

    print("=" * 50)

    print("\nDATASET SUMMARY")

    print(f"Users        : {len(users)}")

    print(f"Movies       : {len(items)}")

    print(
        f"Interactions : {len(interactions)}"
    )


if __name__ == "__main__":
    main()