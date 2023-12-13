from zenml.steps import BaseParameters


class ModelConfig(BaseParameters):
    """
    Model Configurations
    """

    model_name: str = "log_reg"
    optimize: bool = False
    params_dict: dict = {"C": 1.0, "solver": "newton-cg", "max_iter": 216}
