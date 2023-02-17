import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
import wandb


class MyLinearSVM(BaseEstimator, ClassifierMixin):

    weight: NDArray
    C: float
    tol: float
    max_iter: int

    def __init__(self, loss: str == 'hinge', C: float = 1.0, tol: float = 1e-4, max_iter: int = 1000):
        """

        :param loss:  Specifies the loss function. 'hinge' is the standard SVM loss (used e.g. by the SVC class) while
        'squared_hinge' is the square of the hinge loss. The combination of ``penalty='l1'`` and ``loss='hinge'`` is
        not supported.
        :param C:  Regularization parameter. The strength of the regularization is inversely proportional to C.
             Must be strictly positive.
        """
        self.C = C
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, X, y=None):
        X['bias'] = 1  # add coefficients for bias
        if all(y.unique() == [0, 1]):
            # replace 0 with -1 for negative class
            y = np.where(y == 1, 1, -1)
        self.weight = np.random.randn(X.shape[-1])
        loss = np.array([self.tol + 1])  # won't be used as an actual loss
        i: int = 0
        while (loss.mean()) > self.tol and i < self.max_iter:
            preds = self.decision_function(X)
            pred_loss = np.maximum(1 - (y * preds), 0)  # element wise multiply
            loss_mask = np.where(pred_loss > 0, 1, 0)
            pred_loss = pred_loss*loss_mask
            log_dict = {}
            log_dict.update({"mis_classified_points": loss_mask.sum()})
            # don't include bias in the norm calculation
            weight_norm: float = np.linalg.norm(self.weight[:-1])
            loss = 0.5*weight_norm**2 + (self.C*pred_loss)
            log_dict.update({"loss": loss.sum()})
            weight_norm_grad = np.full(self.weight.shape, fill_value=weight_norm)
            # bias does not get weight_norm_grad applied
            weight_norm_grad[-1] = 0
            grads = (self.C*(-y[:, None] * X)*loss_mask[:, None]) + weight_norm_grad
            log_dict.update({f"grads_mean_{k}": v for k, v in grads.mean().items()})

            # log_dict.update({f"grads_std_{k}": v for k, v in grads.std().items()})
            if wandb.run:
                wandb.log(log_dict)

            # finally, apply the update
            self.weight = self.weight - 5e-4*grads.mean(0)
            i += 1


    def predict(self, X, y=None):
        # X['bias'] = 1  # add coefficients for bias
        # weights dotted into features is same as input (rows) mat mul w/ weight vector (column)
        preds = self.decision_function(X)
        return np.where(preds > 0, 1, -1)


    def predict_proba(self, X, y=None):
        pass

    def decision_function(self, X):
        if X.shape[-1] < self.weight.shape[-1]:
            biases = np.ones((X.shape[0], 1))
            X = np.concatenate((X, biases), axis=1)
        return X @ self.weight



if __name__ == '__main__':
    # store the data
    iris = load_iris()

    # convert to DataFrame
    df = pd.DataFrame(data=iris.data,
                      columns=iris.feature_names)

    # store mapping of targets and target names
    target_dict = dict(zip(set(iris.target), iris.target_names))

    # add the target labels and the feature names
    df["target"] = iris.target
    df["target_names"] = df.target.map(target_dict)

    X = df.query("target_names == 'setosa' or target_names == 'versicolor'").loc[:,
        "petal length (cm)":"petal width (cm)"]
    y = df.query("target_names == 'setosa' or target_names == 'versicolor'").loc[:, "target"]
    wandb.init(entity="kingb12", project="svm_from_scratch")
    svc = MyLinearSVM(loss="hinge", C=10)
    svc.fit(X, y)
    svc.predict(X)
