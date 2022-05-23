## Source Code Practice for logistic regression
# optimization objective: maximum likelihood estimation
# optimization method: gradient descent using analytical gradient

from cmath import exp
from logging import lastResort
import numpy as np
import pandas as pd
x_train = np.array([[0.1, 0.2, 1],
                    [0.2, 1, 0.5],
                    [2, 0.2, 2],
                    [0.5, 1, 0.5]]) 

y_train = np.array([0,0,1,1])

class logistic_regression:
    def __init__(self) -> None:
        pass

    def initialize_beta(self, x):
        # beta: all zeros (inclusive of intercept beta0)
        return np.array([0]*(1+x.shape[1]))

    def evaluate_gradient(self, x, y, yhat):   
        error = y-yhat
        gradient_beta_0 = -np.mean(error)
        gradient_beta_x=list(np.apply_along_axis(np.mean, axis = 1, arr = -x.T*(error)))
        gradient_beta = np.array([gradient_beta_0] + gradient_beta_x)
        return gradient_beta

    # in this case, use iterations; can also use rss movement criteria 
    def estimation_gradient_descent(self, x, y, lr = 0.01, iterations = 100):
        all_ones = np.ones((x.shape[0],1))
        x_matrix = np.hstack((all_ones, x))

        beta = self.initialize_beta(x)
        z = np.matmul(x_matrix, beta)
        yhat = 1/(1+np.exp(-z)) # sigmoid/logit transformation: g(z)

        for i in range(iterations):
            gradient_beta = self.evaluate_gradient(x,y, yhat)
            beta = beta-lr*gradient_beta
            z = np.matmul(x_matrix, beta)
            yhat = 1/(1+np.exp(-z))

        fitted = pd.DataFrame({'class': y, 'probability': yhat})

        return beta, fitted

lr = logistic_regression()
beta_logistic_regression, fitted = lr.estimation_gradient_descent(x_train,y_train)


# evaluation metrics for classification problem
# confusion matrix
# 