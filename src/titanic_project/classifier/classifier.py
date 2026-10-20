#==========================================================================================
# IMPORTS & MODULES
#==========================================================================================

import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report, confusion_matrix, accuracy_score, 
    precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split

#==========================================================================================
# LOGISTIC REGRESSION: Custom Model
#==========================================================================================

class TitanicClassifier:

    __slots__ = ("data_path", "dataframe", "model", "metrics", "is_trained")

    def __init__(
        self, 
        data_path: str | Path,
    ) -> TitanicClassifier:
        self.data_path = Path(data_path)
        self.dataframe = None
        self.model = None
        self.metrics = {}
        self.is_trained = False

    def setup(self):
        pass

    def predict(self) -> None:
        pass

    def _load_data(self) -> None:
        pass

    def _process_data(self) -> None:
        pass

    def _fit_model(self) -> None:
        pass