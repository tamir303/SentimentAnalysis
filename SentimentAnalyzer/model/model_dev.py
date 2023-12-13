import optuna
from abc import ABC, abstractmethod
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class Model(ABC):
    """
    Abstract base class for all models.
    """

    @abstractmethod
    def train( self, X_train, y_train ):
        """
        Trains the model on the given data.

        Args:
            X_train: Training data
            y_train: Target data
        """
        pass

    @abstractmethod
    def optimize( self, trial, X_train, X_test, y_train, y_test ):
        """
        Optimizes the hyperparameters of the model.

        Args:
            trial: Optuna trial object
            X_train: Training data
            y_train: Target data
            X_test: Testing data
            y_test: Testing target
        """
        pass


class LogisticRegressionModel(Model):
    """
    LinearRegressionModel that implements the Model interface.
    """

    def train( self, X_train, y_train, **kwargs ):
        # logistic regression
        model = LogisticRegression(**kwargs)
        model.fit(X_train, y_train)

        return model

    def optimize( self, trial, X_train, X_test, y_train, y_test ):
        c = trial.suggest_float("C", 1e-1, 1, step=0.1)
        solver = trial.suggest_categorical("solver", [ "newton-cg", "lbfgs", "liblinear", "sag", "saga" ])
        max_iter = trial.suggest_int("max_iter", 100, 300)
        log_reg = self.train(X_train, y_train, C=c, solver=solver, max_iter=max_iter)

        return log_reg.score(X_test, y_test)


class XGBClassifierModel(Model):
    """
    XGBClassifierModel that implements the Model interface.
    """

    def train( self, X_train, y_train, **kwargs ):
        # XGBClassifier
        model = XGBClassifier(**kwargs)
        model.fit(X_train, y_train)

        return model

    def optimize( self, trial, X_train, X_test, y_train, y_test ):
        n_estimators = trial.suggest_int("n_estimators", 1, 200)
        max_depth = trial.suggest_int("max_depth", 1, 30)
        learning_rate = trial.suggest_loguniform("learning_rate", 1e-7, 10.0)
        reg = self.train(X_train, y_train, n_estimators=n_estimators,
                         learning_rate=learning_rate, max_depth=max_depth)
        return reg.score(X_test, y_test)


class HyperparameterTuner:
    """
    Class for performing hyperparameter tuning. It uses Model strategy to perform tuning.
    """

    def __init__( self, model, X_train, X_test, y_train, y_test ):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def optimize( self, n_trials=100 ):
        """
        Learns best params for a specific model (Fine-Tuning)
        :return:
        """
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial:
                       self.model.optimize(trial, self.X_train, self.X_test, self.y_train, self.y_test),
                       n_trials=n_trials)
        return study.best_trial.params
