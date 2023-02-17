import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

from data import load_pulsar_data


def plot_svm(svc, X, y):
    # https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
    # getting the decision function
    decision_function = svc.decision_function(X)
    support_vector_indices = np.where((2 * y - 1) * decision_function <= 1)[0]
    support_vectors = X.iloc[support_vector_indices]

    # plot observations
    plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # creating the grid to evaluate the model
    xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                         np.linspace(ylim[0], ylim[1], 50))
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # plot decision boundaries and margins
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

    # plot support vectors
    plt.scatter(support_vectors.iloc[:, 0], support_vectors.iloc[:, 1], s=100,
                linewidth=1, facecolors='none', edgecolors='k')

    plt.title("Linear SVM (Hard Margin Classification)")
    plt.tight_layout()
    plt.show()


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

    svc = LinearSVC(loss="hinge", C=1000)
    svc.fit(X, y)

    plot_svm(svc, X, y)
    print(svc.coef_)

if __name__ == '__main__':
    main()