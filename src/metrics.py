import numpy as np


class Metrics:
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float: ...

    def __repr__(self):
        return self.__class__.__name__


class Accuracy(Metrics):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Simple metric for classification tasks"""
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        accuracy = TanhAccuracy() if y_true.min() < 0 else SigmoidAccuracy()
        return accuracy(y_pred, y_true)


class SigmoidAccuracy(Accuracy):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        pred_labels = np.round(y_pred).astype(int)
        true_labels = y_true.astype(int)
        return float(np.mean(pred_labels == true_labels))


class TanhAccuracy(Accuracy):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        pred_labels = (y_pred > 0).astype(int)
        true_labels = (y_true > 0).astype(int)
        return float(np.mean(pred_labels == true_labels))


class R2(Metrics):
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """R-squared metric for regression tasks"""
        y_pred = y_pred.flatten()
        y_true = y_true.flatten()

        with np.errstate(all="ignore"):
            ss_res = np.nansum((y_true - y_pred) ** 2)
            ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return np.clip(r2, -1, 1)
