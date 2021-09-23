from sklearn.base import BaseEstimator


class BaselineModel(BaseEstimator):
	def __init__(self, most_common):
		self.most_common = most_common

	def predict(self, X):
		return [self.most_common] * len(X)

	def fit(self, X, y):
		return self
