from pathlib import Path
import pandas as pd


# -----------------------------------------------------------
# 1. Define project folders
# -----------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ERROR_DIR = PROJECT_ROOT / "data" / "errors"
REPORT_DIR = PROJECT_ROOT / "reports" / "logs"

# Make sure output directories exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ERROR_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------
# 2. Load ingested CSV files
# -----------------------------------------------------------

def load_data():

    print("Loading ingested datasets...")

    users = pd.read_csv(RAW_DIR / "users.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")
    interactions = pd.read_csv(
        RAW_DIR / "interactions.csv"
    )

    print(f"Users loaded        : {len(users)}")
    print(f"Movies loaded       : {len(items)}")
    print(f"Interactions loaded : {len(interactions)}")

    return users, items, interactions


# -----------------------------------------------------------
# 3. Validate required columns
# -----------------------------------------------------------

def validate_required_columns(users, items, interactions):

    print("\nChecking required columns...")

    required_user_columns = {
        "user_id",
        "age",
        "gender",
        "occupation",
        "zip_code"
    }

    required_item_columns = {
        "movie_id",
        "movie_title"
    }

    required_interaction_columns = {
        "user_id",
        "movie_id",
        "rating",
        "timestamp"
    }

    user_missing = required_user_columns - set(users.columns)
    item_missing = required_item_columns - set(items.columns)

    interaction_missing = (
        required_interaction_columns
        - set(interactions.columns)
    )

    if user_missing:
        raise ValueError(
            f"Users missing columns: {user_missing}"
        )

    if item_missing:
        raise ValueError(
            f"Items missing columns: {item_missing}"
        )

    if interaction_missing:
        raise ValueError(
            "Interactions missing columns: "
            f"{interaction_missing}"
        )

    print("Required column check: PASSED")


# -----------------------------------------------------------
# 4. Validate users
# -----------------------------------------------------------

def validate_users(users):

    print("\nValidating users...")

    invalid_mask = pd.Series(
        False,
        index=users.index
    )

    # Missing values in important user fields
    missing_mask = users[
        [
            "user_id",
            "age",
            "gender",
            "occupation"
        ]
    ].isnull().any(axis=1)

    # Duplicate user IDs
    duplicate_mask = users.duplicated(
        subset=["user_id"],
        keep=False
    )

    # Age must be reasonable
    invalid_age_mask = (
        (users["age"] <= 0)
        | (users["age"] > 120)
    )

    invalid_mask = (
        missing_mask
        | duplicate_mask
        | invalid_age_mask
    )

    invalid_users = users[invalid_mask].copy()
    valid_users = users[~invalid_mask].copy()

    print(
        f"Valid users   : {len(valid_users)}"
    )

    print(
        f"Invalid users : {len(invalid_users)}"
    )

    return valid_users, invalid_users


# -----------------------------------------------------------
# 5. Validate movies/items
# -----------------------------------------------------------

def validate_items(items):

    print("\nValidating movies...")

    # movie_id and title cannot be missing
    missing_mask = items[
        ["movie_id", "movie_title"]
    ].isnull().any(axis=1)

    # Movie IDs should be unique
    duplicate_mask = items.duplicated(
        subset=["movie_id"],
        keep=False
    )

    invalid_mask = (
        missing_mask
        | duplicate_mask
    )

    invalid_items = items[invalid_mask].copy()
    valid_items = items[~invalid_mask].copy()

    print(
        f"Valid movies   : {len(valid_items)}"
    )

    print(
        f"Invalid movies : {len(invalid_items)}"
    )

    return valid_items, invalid_items


# -----------------------------------------------------------
# 6. Validate interactions
# -----------------------------------------------------------

def validate_interactions(
    interactions,
    valid_users,
    valid_items
):

    print("\nValidating interactions...")

    # We will record why each row failed
    validation = interactions.copy()

    validation["error_reason"] = ""

    # -----------------------------
    # Missing values
    # -----------------------------

    missing_mask = validation[
        [
            "user_id",
            "movie_id",
            "rating",
            "timestamp"
        ]
    ].isnull().any(axis=1)

    validation.loc[
        missing_mask,
        "error_reason"
    ] += "Missing value; "

    # -----------------------------
    # Duplicate interaction
    # -----------------------------

    duplicate_mask = validation.duplicated(
        subset=[
            "user_id",
            "movie_id",
            "timestamp"
        ],
        keep=False
    )

    validation.loc[
        duplicate_mask,
        "error_reason"
    ] += "Duplicate interaction; "

    # -----------------------------
    # Rating validation
    # MovieLens uses ratings 1-5
    # -----------------------------

    rating_numeric = pd.to_numeric(
        validation["rating"],
        errors="coerce"
    )

    invalid_rating_mask = (
        rating_numeric.isna()
        | ~rating_numeric.between(1, 5)
    )

    validation.loc[
        invalid_rating_mask,
        "error_reason"
    ] += "Invalid rating; "

    # -----------------------------
    # Timestamp validation
    # -----------------------------

    timestamp_numeric = pd.to_numeric(
        validation["timestamp"],
        errors="coerce"
    )

    invalid_timestamp_mask = (
        timestamp_numeric.isna()
        | (timestamp_numeric <= 0)
    )

    validation.loc[
        invalid_timestamp_mask,
        "error_reason"
    ] += "Invalid timestamp; "

    # -----------------------------
    # User referential integrity
    # -----------------------------

    invalid_user_mask = ~validation[
        "user_id"
    ].isin(valid_users["user_id"])

    validation.loc[
        invalid_user_mask,
        "error_reason"
    ] += "Unknown user_id; "

    # -----------------------------
    # Movie referential integrity
    # -----------------------------

    invalid_movie_mask = ~validation[
        "movie_id"
    ].isin(valid_items["movie_id"])

    validation.loc[
        invalid_movie_mask,
        "error_reason"
    ] += "Unknown movie_id; "

    # -----------------------------
    # Separate valid / invalid
    # -----------------------------

    invalid_mask = (
        validation["error_reason"] != ""
    )

    invalid_interactions = (
        validation[invalid_mask].copy()
    )

    valid_interactions = (
        validation[~invalid_mask]
        .drop(columns=["error_reason"])
        .copy()
    )

    print(
        f"Valid interactions   : "
        f"{len(valid_interactions)}"
    )

    print(
        f"Invalid interactions : "
        f"{len(invalid_interactions)}"
    )

    return (
        valid_interactions,
        invalid_interactions
    )


# -----------------------------------------------------------
# 7. Save validation results
# -----------------------------------------------------------

def save_results(
    valid_users,
    invalid_users,
    valid_items,
    invalid_items,
    valid_interactions,
    invalid_interactions
):

    print("\nSaving validation outputs...")

    # Valid datasets
    valid_users.to_csv(
        PROCESSED_DIR / "valid_users.csv",
        index=False
    )

    valid_items.to_csv(
        PROCESSED_DIR / "valid_items.csv",
        index=False
    )

    valid_interactions.to_csv(
        PROCESSED_DIR / "valid_interactions.csv",
        index=False
    )

    # Invalid datasets
    invalid_users.to_csv(
        ERROR_DIR / "invalid_users.csv",
        index=False
    )

    invalid_items.to_csv(
        ERROR_DIR / "invalid_items.csv",
        index=False
    )

    invalid_interactions.to_csv(
        ERROR_DIR / "invalid_interactions.csv",
        index=False
    )

    print("Validation datasets saved.")


# -----------------------------------------------------------
# 8. Create validation report
# -----------------------------------------------------------

def create_report(
    users,
    items,
    interactions,
    valid_users,
    invalid_users,
    valid_items,
    invalid_items,
    valid_interactions,
    invalid_interactions
):

    report_file = (
        REPORT_DIR / "validation_report.txt"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as report:

        report.write(
            "DATA VALIDATION REPORT\n"
        )

        report.write(
            "=" * 50 + "\n\n"
        )

        report.write("USERS\n")
        report.write("-" * 30 + "\n")

        report.write(
            f"Total users   : {len(users)}\n"
        )

        report.write(
            f"Valid users   : {len(valid_users)}\n"
        )

        report.write(
            f"Invalid users : {len(invalid_users)}\n\n"
        )

        report.write("MOVIES\n")
        report.write("-" * 30 + "\n")

        report.write(
            f"Total movies   : {len(items)}\n"
        )

        report.write(
            f"Valid movies   : {len(valid_items)}\n"
        )

        report.write(
            f"Invalid movies : {len(invalid_items)}\n\n"
        )

        report.write("INTERACTIONS\n")
        report.write("-" * 30 + "\n")

        report.write(
            "Total interactions   : "
            f"{len(interactions)}\n"
        )

        report.write(
            "Valid interactions   : "
            f"{len(valid_interactions)}\n"
        )

        report.write(
            "Invalid interactions : "
            f"{len(invalid_interactions)}\n\n"
        )

        report.write("VALIDATION RULES\n")
        report.write("-" * 30 + "\n")

        report.write(
            "1. Required columns checked\n"
        )

        report.write(
            "2. Missing values checked\n"
        )

        report.write(
            "3. Duplicate records checked\n"
        )

        report.write(
            "4. User age checked\n"
        )

        report.write(
            "5. Rating range 1-5 checked\n"
        )

        report.write(
            "6. Timestamp validity checked\n"
        )

        report.write(
            "7. User referential integrity checked\n"
        )

        report.write(
            "8. Movie referential integrity checked\n"
        )

    print(
        f"Validation report created: {report_file}"
    )


# -----------------------------------------------------------
# 9. Main validation pipeline
# -----------------------------------------------------------

def main():

    print("=" * 55)
    print("DATA VALIDATION STARTED")
    print("=" * 55)

    users, items, interactions = load_data()

    validate_required_columns(
        users,
        items,
        interactions
    )

    (
        valid_users,
        invalid_users
    ) = validate_users(users)

    (
        valid_items,
        invalid_items
    ) = validate_items(items)

    (
        valid_interactions,
        invalid_interactions
    ) = validate_interactions(
        interactions,
        valid_users,
        valid_items
    )

    save_results(
        valid_users,
        invalid_users,
        valid_items,
        invalid_items,
        valid_interactions,
        invalid_interactions
    )

    create_report(
        users,
        items,
        interactions,
        valid_users,
        invalid_users,
        valid_items,
        invalid_items,
        valid_interactions,
        invalid_interactions
    )

    print("\n" + "=" * 55)

    print(
        "DATA VALIDATION COMPLETED SUCCESSFULLY"
    )

    print("=" * 55)

    print("\nFINAL VALIDATION SUMMARY")

    print(
        f"Users        : "
        f"{len(valid_users)} valid, "
        f"{len(invalid_users)} invalid"
    )

    print(
        f"Movies       : "
        f"{len(valid_items)} valid, "
        f"{len(invalid_items)} invalid"
    )

    print(
        f"Interactions : "
        f"{len(valid_interactions)} valid, "
        f"{len(invalid_interactions)} invalid"
    )


if __name__ == "__main__":
    main()