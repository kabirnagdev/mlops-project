
from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = "mp033-canteen-champion"

model = client.get_registered_model(model_name)

print("Registered model:", model.name)
print("\nRegistered versions:")

versions = client.search_model_versions(
    f"name='{model_name}'"
)

for version in versions:
    print(
        "Version:", version.version,
        "| Run ID:", version.run_id,
        "| Status:", version.status
    )

champion = client.get_model_version_by_alias(
    model_name,
    "champion"
)

print("\nChampion alias:")
print("Alias: champion")
print("Version:", champion.version)
print("Run ID:", champion.run_id)
