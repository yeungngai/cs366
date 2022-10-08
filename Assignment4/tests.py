from utils import plot_data, generate_data
import numpy as np

from sklearn.linear_model import LogisticRegression

"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""


def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """

    print(X)
    print(t)
    print(len(X))
    print((len(t)))

    model = LogisticRegression(solver='liblinear', random_state=0)
    model.fit(X, t)
    wfunc = model.coef_[0]
    bfunc = model.intercept_[0]



    weight = [0 for x in range(len(X[0]))]
    b = 0
    for i in range(1000):
        dw = [0 for x in range(len(X[0]))]
        db = 0
        for j in range(len(t)):
            zj = np.dot(weight, X[j]) + b
            yj = 1 / (1 + np.exp(-zj))
            dw = dw + ((yj - t[j]) * X[j])
            db = db + (yj - t[j])
        weight = weight - 0.1*dw
        b = b - 0.1*db

    w = weight

    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """

    t = []
    for i in range(len(X)):
        ti = np.dot(w, X[i]) + b
        ti = 1 / (1 + np.exp(-ti))
        t.append(ti)

    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """

    weight = [0 for x in range(len(X[0]))]
    b = 0

    for i in range(1000):
        dw = np.dot(np.transpose(X), np.dot(X, weight) - t) * (1/len(t))
        db = (np.dot(X, weight) - t) * (1 / len(t))

        weight = weight - 0.1*dw
        b = b - 0.1*db

    w = weight
    b = np.average(b)

    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    t = np.dot(X, w) + b

    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    sum = 0
    for i in range(len(t)):
        temp = np.square(t_hat[i] - t[i])
        sum = sum + temp

    return 1 - (sum / len(t))


def main():
    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
