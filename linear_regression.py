import numpy as np

class Predictor(object):
    def linear_regression(self, mat_X, vec_Y):
        """
        Learn the parameter \vec{a} for the linear regression model
        """
        mat_A = np.dot(mat_X, mat_X.T)
        vec_b = np.dot(mat_X, vec_Y.T)
        vec_a = np.dot(np.linalg.inv(mat_A), vec_b)
        return vec_a
    
    def predict(vec_a, mat_X):
        vec_Y_predicted = np.dot(vec_a.T, mat_X)
        return vec_Y_predicted
