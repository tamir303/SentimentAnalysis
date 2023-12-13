import json
import logging
import pickle

import numpy as np
import pandas as pd
from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml import step, pipeline
from data.get_data_for_test import get_data_for_test
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.services import BaseService
from zenml.integrations.mlflow.services.mlflow_deployment import (
    MLFlowDeploymentService,
)

docker_settings = DockerSettings(required_integrations=[ MLFLOW ])


@step(enable_cache=True)
def dynamic_import() -> str:
    data = get_data_for_test()
    return data


@step(enable_cache=False)
def prediction_service_loader(
        pipeline_name: str,
        pipeline_step_name: str,
        running: bool = True,
        model_name: str = "model",
        timeout: int = 10
) -> BaseService:
    """Get the prediction service started by the deployment pipeline.

    Args:
        pipeline_name: name of the pipeline that deployed the MLFlow prediction
            server
        pipeline_step_name: the name of the step that deployed the MLFlow prediction
            server
        running: when this flag is set, the step only returns a running service
        model_name: the name of the model that is deployed
        timeout
    """

    model_deployer = MLFlowModelDeployer.get_active_model_deployer()
    services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running,
    )

    if services:
        if services[ 0 ].is_running:
            print(
                f"Model deployment service started and reachable at:\n"
                f"    {services[ 0 ].prediction_url}\n"
            )
        elif services[ 0 ].is_failed:
            raise RuntimeError(
                f"No MLflow prediction service deployed by the "
                f"{pipeline_step_name} step in the {pipeline_name} "
                f"pipeline for the '{model_name}' model is currently "
                f"running."
            )

        services[ 0 ].start(timeout=timeout)

    print(services[ 0 ])
    print(type(services[ 0 ]))

    return services[ 0 ]


@step
def predictor(
        service: MLFlowDeploymentService,
        data: str,
        label_predictions: bool = True
) -> np.ndarray:
    """Run an inference request against a prediction service"""
    try:
        data = json.loads(data)
        parsed_data = np.concatenate([np.array(text, dtype=str) for text in data["data"]])
        prediction = service.predict(parsed_data)

        if label_predictions:
            with open("save_model/model.pkl", "rb") as file:
                lvm = pickle.load(file)
            labels = lvm.le.inverse_transform(prediction)

            logging.info("Predictions:")
            for pair in zip(parsed_data, labels):
                logging.info(pair)

        return prediction

    except Exception as e:
        logging.error(e)
        raise e


@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_pipeline( pipeline_name: str, pipeline_step_name: str ):
    # Link all the steps artifacts together
    batch_data = dynamic_import()
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )

    prediction = predictor(service=model_deployment_service, data=batch_data)
