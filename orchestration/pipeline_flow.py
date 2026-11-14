from pathlib import Path
import subprocess
import sys
from datetime import datetime

from prefect import flow, task


# ============================================================
# PROJECT SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_python_script(script_path):
    """
    Run one pipeline Python script.

    If the script fails, the complete Prefect pipeline
    also fails.
    """

    full_path = PROJECT_ROOT / script_path

    print(f"\nRunning: {script_path}")
    print("-" * 60)

    result = subprocess.run(
        [
            sys.executable,
            str(full_path)
        ],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True
    )

    # Show the output from the individual script
    if result.stdout:
        print(result.stdout)

    # If the script fails, display the error and stop
    if result.returncode != 0:

        if result.stderr:
            print(result.stderr)

        raise RuntimeError(
            f"Pipeline stage failed: {script_path}"
        )

    print(
        f"Successfully completed: {script_path}"
    )

    return True


# ============================================================
# PIPELINE TASK 1 - INGESTION
# ============================================================

@task(name="Data Ingestion")
def ingestion_task():

    return run_python_script(
        "src/ingestion/ingest_data.py"
    )


# ============================================================
# PIPELINE TASK 2 - VALIDATION
# ============================================================

@task(name="Data Validation")
def validation_task():

    return run_python_script(
        "src/validation/validate_data.py"
    )


# ============================================================
# PIPELINE TASK 3 - PREPARATION
# ============================================================

@task(name="Data Preparation")
def preparation_task():

    return run_python_script(
        "src/preparation/prepare_data.py"
    )


# ============================================================
# PIPELINE TASK 4 - FEATURE MANAGEMENT
# ============================================================

@task(name="Feature Management")
def feature_task():

    return run_python_script(
        "src/features/build_features.py"
    )


# ============================================================
# PIPELINE TASK 5 - MODEL TRAINING
# ============================================================

@task(name="Model Training")
def training_task():

    return run_python_script(
        "src/training/train_model.py"
    )


# ============================================================
# PIPELINE TASK 6 - MODEL EVALUATION
# ============================================================

@task(name="Model Evaluation")
def evaluation_task():

    return run_python_script(
        "src/evaluation/evaluate_models.py"
    )


# ============================================================
# COMPLETE PIPELINE FLOW
# ============================================================

@flow(
    name="Movie Recommendation Data Management Pipeline",
    log_prints=True
)
def recommendation_pipeline():

    print("=" * 70)
    print("END-TO-END RECOMMENDATION PIPELINE STARTED")
    print("=" * 70)

    start_time = datetime.now()

    # Each task waits for the previous task.
    ingestion_future = ingestion_task.submit()

    validation_future = validation_task.submit(
        wait_for=[ingestion_future]
    )

    preparation_future = preparation_task.submit(
        wait_for=[validation_future]
    )

    feature_future = feature_task.submit(
        wait_for=[preparation_future]
    )

    training_future = training_task.submit(
        wait_for=[feature_future]
    )

    evaluation_future = evaluation_task.submit(
        wait_for=[training_future]
    )

    # Wait until the final stage finishes.
    evaluation_future.result()

    end_time = datetime.now()

    duration = end_time - start_time

    print("\n" + "=" * 70)
    print(
        "END-TO-END RECOMMENDATION PIPELINE "
        "COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(f"\nPipeline duration: {duration}")


# ============================================================
# RUN THE PIPELINE
# ============================================================

if __name__ == "__main__":
    recommendation_pipeline()