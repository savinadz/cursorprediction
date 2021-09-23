from sklearn.base import BaseEstimator
import config


class DistanceModel(BaseEstimator):
	""" This models predicts always the class from which the anchor point is the closest to the cursor. """

	def __init__(self):
		pass

	def predict(self, X):
		y_pred = []
		for x in X:
			x_ = list(x[:7])
			min_val = min(x_)
			min_val_index = x_.index(min_val)
			y_pred.append(config.targets[min_val_index])

		return y_pred

	def fit(self, X, y):
		return self