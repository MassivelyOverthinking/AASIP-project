# -----------------------------------------------------------------------------------------------------------------------
# IMPORTS & MODULES
# -----------------------------------------------------------------------------------------------------------------------

import pandas as pd

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------------------------------------------------
# LOGISTIC REGRESSION: Custom Model
# -----------------------------------------------------------------------------------------------------------------------

class TitanicClassifier:

    __slots__ = ("data_path", "dataframe", "model", "_diagnostics", "_cm_matrix", "is_trained")

    def __init__(
        self, 
        data_path: str | Path,
    ) -> TitanicClassifier:
        self.data_path = Path(data_path)
        self.dataframe = None
        self.model = None
        self._diagnostics = {}
        self._cm_matrix = None
        self.is_trained = False

        self._setup(self.data_path)

    def _setup(self, filepath: str | Path) -> None:
        if not isinstance(filepath, (str, Path)):
            raise TypeError(f"Filepath must be Type: Str | Path - Found dtype: {type(filepath).__name__}")

        filepath = Path(filepath)

        self._load_data(filepath)
        X_train, X_test, y_train, y_test = self._process_data()
        y_pred = self._fit_model(X_train, X_test, y_train)
        self._get_diagnostics(y_pred, y_test)

    def predict(self, passengers: list[dict]) -> list[dict]:
        if not self.is_trained:
            raise RuntimeError("Classifier has not been trained.")

        if not passengers:
            return []

        features = pd.DataFrame(passengers)

        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)[:, 1]

        return [
            {
                "survived": int(pred),
                "survival_probability": float(prob),
            }
            for pred, prob in zip(
                predictions, probabilities
            )
        ]

    @property
    def diagnostics(self) -> dict[str, float]:
        if self._diagnostics is None:
            raise ValueError("No diagnostics found!")

        return self._diagnostics.copy()

    @property
    def confusion_matrix(self) -> any:
        if self._cm_matrix is None:
            raise ValueError("No matrix found!")
    
        return self._cm_matrix.copy()

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

        self.model = LogisticRegression(
            max_iter=2000, 
            random_state=42
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        y_pred = self.model.predict(X_test)  

        return y_pred

    def _get_diagnostics(self, y_pred, y_test):

        diag_dict: dict[str, float] = {}
        diag_dict.update({
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        })

        self._cm_matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
        self._diagnostics = diag_dict
