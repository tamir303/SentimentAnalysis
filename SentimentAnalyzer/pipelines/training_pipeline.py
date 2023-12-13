from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(name="train_pipeline", settings={"docker": docker_settings})
def train_pipeline(ingest_data, prepare_data, model_train, model_eval):
    """
    Args:
        ingest_data: DataClass
        prepare_data: DataClass
        model_train: DataClass
        model_eval: DataClass
    Returns:
        mse: float
        rmse: float
    """
    train, val = ingest_data()
    X_train, X_test, y_train, y_test, val = prepare_data(train, val)
    model = model_train(X_train, X_test, y_train, y_test)
    mse, rmse = model_eval(model, val)

