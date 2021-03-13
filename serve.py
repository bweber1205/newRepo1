from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
import os
import mlflow.sklearn
import pandas as pd

run = MlflowClient().search_runs(
  experiment_ids="0",
  filter_string="",
  run_view_type=ViewType.ACTIVE_ONLY,
  max_results=1,
  order_by=["metrics.rmse DESC"]
)[0]


#Load model
model = mlflow.sklearn.load_model(run.info.artifact_uri + "/model/")
print(run.info.artifact_uri + "/model/")
#Inference
wine_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wine-quality.csv")
data = pd.read_csv(wine_path)
test = data.drop(["quality"], axis=1)
print(model.predict(test))