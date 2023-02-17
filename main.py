import pandas as pd
import wandb
from sklearn.datasets import load_iris

from data import load_pulsar_data
from sklearn_comparison import plot_svm
from svm import MyLinearSVM


def main():
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

    # fit the model with hard margin (Large C parameter)
    X = X.fillna(0)  # hack that may not be suitable for a real problem

    wandb.init(entity="kingb12", project="svm_from_scratch")
    svc = MyLinearSVM(loss="hinge", C=1000)
    svc.fit(X, y)

    plot_svm(svc, X, y)
    print(svc.weight)
    

if __name__ == '__main__':
    main()