
import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "mp033-canteen-champion"

# Actual run IDs from your comparison output
MODEL_RUNS = {
    "random_forest": "75efa5c58841405b8ec56bb3eed3ae37",
    "gradient_boosting": "57130c635d4a4f89b352d20599f58566",
}


def register_or_get_version(client, model_name, run_id):
    """Register a run's model once and return its model version."""

    existing_versions = client.search_model_versions(
        f"name='{model_name}'"
    )

    for version in existing_versions:
        if version.run_id == run_id:
            print(
                f"Already registered: version {version.version}, "
                f"run {run_id}"
            )
            return version.version

    model_uri = f"runs:/{run_id}/model"

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name,
    )

    print(
        f"Registered version {registered.version} "
        f"from run {run_id}"
    )

    return registered.version


def main():
    client = MlflowClient()

    # Create registered model if it does not exist
    try:
        client.get_registered_model(MODEL_NAME)
        print("Registered model already exists:", MODEL_NAME)

    except Exception:
        client.create_registered_model(MODEL_NAME)
        print("Created registered model:", MODEL_NAME)

    # Register both model versions
    rf_version = register_or_get_version(
        client,
        MODEL_NAME,
        MODEL_RUNS["random_forest"],
    )

    gb_version = register_or_get_version(
        client,
        MODEL_NAME,
        MODEL_RUNS["gradient_boosting"],
    )

    # Wait until both versions are ready
    for version in [rf_version, gb_version]:
        client.get_model_version(
            name=MODEL_NAME,
            version=version,
        )

    # Random Forest had the lowest observed validation MAE
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias="champion",
        version=rf_version,
    )

    print("\nModel registration completed")
    print("--------------------------------")
    print("Registered model:", MODEL_NAME)
    print("Random Forest version:", rf_version)
    print("Gradient Boosting version:", gb_version)
    print("Champion version:", rf_version)
    print("Champion alias: @champion")


if __name__ == "__main__":
    main()
