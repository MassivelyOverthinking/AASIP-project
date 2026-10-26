# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS & MODULES
# -----------------------------------------------------------------------------------------------------------------------

import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION: Custom Model
# -----------------------------------------------------------------------------------------------------------------------

class TitanicClassifier:

    __slots__ = (
        "data_path", 
        "dataframe", 
        "classifier", 
        "mlp", 
        "_classifier_diagnostics", 
        "_classifier_matrix", 
        "_mlp_diagnostics", 
        "_mlp_matrix",
        "is_trained"
    )

    def __init__(
        self, 
        data_path: str | Path,
    ) -> TitanicClassifier:
        self.data_path = Path(data_path)
        self.dataframe = None
        self.classifier = None
        self.mlp = None
        self._classifier_diagnostics = {}
        self._classifier_matrix = None
        self._mlp_diagnostics = {}
        self._mlp_matrix = None
        self.is_trained = False

        self._setup(self.data_path)

    def _setup(self, filepath: str | Path) -> None:
        if not isinstance(filepath, (str, Path)):
            raise TypeError(f"Filepath must be Type: Str | Path - Found dtype: {type(filepath).__name__}")

        filepath = Path(filepath)

        self._load_data(filepath)
        X_train, X_test, y_train, y_test = self._process_data()
        y_pred_cls, y_pred_mlp = self._fit_model(X_train, X_test, y_train)
        self._get_diagnostics(y_pred_cls, y_pred_mlp, y_test)

    def predict_cls(self, passenger: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained.")

        if not passenger:
            raise RuntimeError("No Passenger data received.")

        features = pd.DataFrame([passenger])

        prediction = self.classifier.predict(features)[0]
        probability = self.classifier.predict_proba(features)[0, 1]

        return {
            "survived": int(prediction),
            "probability": float(probability),
        }

    def predict_mlp(self, passenger: dict) -> dict:
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained.")
    
        if not passenger:
            raise RuntimeError("No Passenger data received.")
    
        features = pd.DataFrame([passenger])
    
        prediction = self.mlp.predict(features)[0]
        probability = self.mlp.predict_proba(features)[0, 1]
    
        return {
            "survived": int(prediction),
            "probability": float(probability),
        }

    @property
    def diagnostics_cls(self) -> dict[str, float]:
        if self._classifier_diagnostics is None:
            raise ValueError("No diagnostics found!")

        return self._classifier_diagnostics.copy()

    @property
    def diagnostics_mlp(self) -> dict[str, float]:
        if self._mlp_diagnostics is None:
            raise ValueError("No diagnostics found!")
    
        return self._mlp_diagnostics.copy()

    @property
    def cls_matrix(self) -> any:
        if self._classifier_matrix is None:
            raise ValueError("No matrix found!")
    
        return self._classifier_matrix.copy().tolist()

    @property
    def mlp_matrix(self) -> any:
        if self._mlp_matrix is None:
            raise ValueError("No matrix found!")
        
        return self._mlp_matrix.copy().tolist()

    def _load_data(self, filepath: Path) -> None:
        self.data_path = filepath
        self.dataframe = pd.read_csv(filepath)

    def _process_data(self) -> None:
        if self.dataframe is None:
            raise ValueError("Internal Dataframe not found.")

        self.dataframe.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Fare", "Embarked"], inplace=True)

        self.dataframe.dropna(inplace=True)

        self.dataframe["Sex"] = self.dataframe["Sex"].map({
            "male": 0,
            "female": 1,
        })

        # Create X and Y test arrays.
        X = self.dataframe.drop(columns=["Survived"])
        y = self.dataframe["Survived"].astype(int)

        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )

        return X_train, X_test, y_train, y_test

    def _fit_model(self, X_train, X_test, y_train):
        if self.is_trained:
            return ValueError("Internal model is already trained.")

        self.classifier = LogisticRegression(
            max_iter=2000, 
            random_state=42
        )

        self.mlp = MLPClassifier(
            max_iter=2000,
            random_state=42
        )

        self.classifier.fit(X_train, y_train)
        self.mlp.fit(X_train, y_train)
        self.is_trained = True

        y_pred_classifier = self.classifier.predict(X_test) 
        y_pred_mlp = self.mlp.predict(X_test) 

        return y_pred_classifier, y_pred_mlp

    def _get_diagnostics(self, y_pred_classifier, y_pred_mlp, y_test):

        classifier_dict: dict[str, float] = {}
        mlp_dict: dict[str, float] = {}

        classifier_dict.update({
            "Accuracy": round(accuracy_score(y_test, y_pred_classifier), 2),
            "Precision": round(precision_score(y_test, y_pred_classifier, zero_division=0), 2),
            "Recall": round(recall_score(y_test, y_pred_classifier, zero_division=0), 2),
            "F1-Score": round(f1_score(y_test, y_pred_classifier, zero_division=0), 2),
        })

        mlp_dict.update({
            "Accuracy": round(accuracy_score(y_test, y_pred_mlp), 2),
            "Precision": round(precision_score(y_test, y_pred_mlp, zero_division=0), 2),
            "Recall": round(recall_score(y_test, y_pred_mlp, zero_division=0), 2),
            "F1-Score": round(f1_score(y_test, y_pred_mlp, zero_division=0), 2),
        })

        self._classifier_matrix = confusion_matrix(y_test, y_pred_classifier, labels=[0, 1])
        self._mlp_matrix = confusion_matrix(y_test, y_pred_mlp, labels=[0, 1])

        self._classifier_diagnostics = classifier_dict
        self._mlp_diagnostics = mlp_dict
