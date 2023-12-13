from zenml import step, pipeline
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.mlflow.steps.mlflow_deployer import (
    mlflow_model_deployer_step,
)
from zenml.integrations.constants import MLFLOW
from zenml.steps import BaseParameters
from steps.ingest_data import ingest_data
from steps.prepare_data import prepare_data
from steps.model_encoders import get_encoders
from steps.model_train import train_model
from steps.model_eval import evaluation
from model.LinearVectorizeModel import LinearVectorizeModel
import pickle

docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    """Parameters that are used to trigger the deployment"""

    min_accuracy: float = 0.92

@step
def deployment_trigger(
        accuracy: float,
        config: DeploymentTriggerConfig
) -> bool:
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy >= config.min_accuracy


class MLFlowDeploymentLoaderStepParameters(BaseParameters):
    """MLFlow deployment getter parameters

    Attributes:
        pipeline_name: name of the pipeline that deployed the MLFlow prediction
            server
        step_name: the name of the step that deployed the MLFlow prediction
            server
        running: when this flag is set, the step only returns a running service
    """

    pipeline_name: str
    step_name: str
    running: bool = True

@step
def save_model(model: LinearVectorizeModel, deployment_decision: bool) -> None:
    if deployment_decision:
        filename = "save_model/model.pkl"
        pickle.dump(model, open(filename, "wb"))

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(
    min_accuracy: float = 0.9,
    workers: int = 1,
    timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT,
):
    # Link all the steps artifacts together
    train, val = ingest_data()
    x_train, x_test, y_train, y_test, val = prepare_data(train, val)
    vectorizer, le = get_encoders(x_train, y_train)
    model = train_model(x_train, x_test, y_train, y_test, vectorizer, le)
    r2, rmse = evaluation(model, val)
    deployment_decision = deployment_trigger(accuracy=r2)
    save_model(model, deployment_decision)
    mlflow_model_deployer_step(
        model=model,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout,
    )






