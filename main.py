import random
import os
import pickle

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from sklearn.tree import export_graphviz
from feature_extractor import FeatureExtractor
from baseline_model import BaselineModel
from distance_model import DistanceModel

import helpers
import analysis
import config

random.seed(10)


if __name__ == '__main__':

	most_common = analysis.analyze(config.experiment_nums, verbose=False)

	cp = FeatureExtractor()
	X_trains, y_trains, X_tests, y_tests = cp.load_experiments_for_training()

	if config.baseline:
		# Naive Baseline
		print('----------Baseline Model----------')
		bl = BaselineModel(most_common)
		helpers.evaluate_leave_one_out(bl, X_trains, y_trains, X_tests, y_tests, labels=None, verbose=False)

	if config.distance_model:
		# Distance Baseline
		print('----------Distance Model----------')
		di = DistanceModel()
		helpers.evaluate_leave_one_out(di, X_trains, y_trains, X_tests, y_tests, labels=None, verbose=False)

	if config.decision_tree:
		# Decision Tree
		print('----------Decision Tree Model----------')
		dt = DecisionTreeClassifier(random_state=42)
		helpers.evaluate_leave_one_out(dt, X_trains, y_trains, X_tests, y_tests, verbose=False)

	if config.random_forest:
		# Random Forest
		print('----------Random Forest Model----------')
		rf = RandomForestClassifier(random_state=42, n_estimators=50)
		helpers.evaluate_leave_one_out(rf, X_trains, y_trains, X_tests, y_tests, labels=cp.label_vector, verbose=False)
		# pickle.dump(rf, open(f'models/random_forest_{config.experiment_nums}.p', 'wb'))

		# Tree images
		#for i, tree in enumerate(rf.estimators_[:3]):
		#	with open(f'tree_imgs/tree_{i}.dot', 'w') as f:
		#		export_graphviz(tree, out_file=f, feature_names=cp.label_vector, filled=True, rounded=True)

		#	os.system(f'dot -Tpng tree_imgs/tree_{i}.dot -o tree_imgs/tree_{i}.png')

	if config.ada_boost:
		# AdaBoost
		print('----------AdaBoost Model----------')
		ab = AdaBoostClassifier(random_state=42)
		helpers.evaluate_leave_one_out(ab, X_trains, y_trains, X_tests, y_tests, labels=cp.label_vector, verbose=False)

	if config.gradient_boost:
		# GradientBoost
		print('----------GradientBoost Model----------')
		gb = GradientBoostingClassifier(random_state=42)
		helpers.evaluate_leave_one_out(gb, X_trains, y_trains, X_tests, y_tests, labels=cp.label_vector, verbose=False)

	if config.logistic_regression:
		# Logistic Regression
		print('----------Logistic Regression Model----------')
		clf = LogisticRegression(max_iter=10000, random_state=42)
		helpers.evaluate_leave_one_out(clf, X_trains, y_trains, X_tests, y_tests, labels=None, verbose=False, scale=True)
