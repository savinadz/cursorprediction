import os
import json
import config
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler, LabelEncoder


""" Provides all relevant methods for training and evaluating """


def load_experiment(session_id, experiment_num, experiment_id):
	""" Given a session ID, the experiment number (1,2,3) and the experiment ID, loads all events from this experiment """
	events = []
	for file in os.listdir(f'data/cursordata/{session_id}/{experiment_num}/{experiment_id}'):
		with open(f'data/cursordata/{session_id}/{experiment_num}/{experiment_id}/{file}', 'r') as f:
			for line in f:
				events.append(json.loads(line))

	return events


def get_onehot(html_id):
	""" Creates a one-hot encoded representation of the HTML element """
	vector = [0] * len(config.targets)
	for i, target in enumerate(config.targets):
		if target in html_id:  # if is substring
			vector[i] = 1
			return target, vector


def save_anchors(session_id):
	""" Calculates anchor points from a given sessionID and stores in json format """

	print(f"Saving anchors from session id {session_id}")

	anchors = []
	with open(f'data/websitedata/targets_{session_id}.jsonl', 'r') as f:
		for line in f:
			anchors.append(json.loads(line))

	anchor_points = {}

	for target in config.anchors:
		anchor_points[target] = (9999999, 9999999)
		for anchor in anchors:
			if target in anchor['id']:
				if (anchor['offset'][0] <= anchor_points[target][0]) & (
						anchor['offset'][1] <= anchor_points[target][1]):
					anchor_points[target] = anchor['offset']

	with open(f'anchors/anchors_{session_id}.json', 'w+') as f:
		json.dump(anchor_points, f)


def load_anchors(session_id):
	""" Loads anchor points from a given sessionID """

	with open(f'anchors/anchors_{session_id}.json', 'r') as f:
		anchors = json.load(f)

	return anchors


def train(clf, X, y):
	# print(f'Using {len(y)} events for training.')
	clf.fit(np.array(X), np.array(y))
	return clf


def evaluate(clf, X, y, labels=None, verbose=False):

	if len(y) == 0:
		# print('No events to evaluate.')
		return 0, 0, 0, 0, 0, np.array([[0] * len(config.targets)] * len(config.targets))

	# print(f'Using {len(y)} events for evaluation.')
	y_pred = clf.predict(np.array(X))

	if verbose and hasattr(clf, 'classes_') and hasattr(clf, 'feature_importances_'):
		print('Classes:', clf.classes_)
		print('Feature importances:', clf.feature_importances_)
		# feature permutation
		# https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

	macro_accuracy = accuracy_score(y, y_pred)

	# for micro accuracy, calculate accuracy for this test session for each target value.
	# Then, calculate the average.

	accuracies = {
		# target: [<correct>, <all>]
		target: [0, 0] for target in config.targets
	}

	for i, val in enumerate(y):
		if val == y_pred[i]:
			accuracies[val][0] += 1

		accuracies[val][1] += 1

	micro_accuracies = [values[0] / values[1] for _, values in accuracies.items() if values[1] > 0]
	micro_accuracy = sum(micro_accuracies) / len(micro_accuracies)

	precision = precision_score(y, y_pred, average='macro', zero_division=0)
	recall = recall_score(y, y_pred, average='macro', zero_division=0)
	f1 = f1_score(y, y_pred, average='macro', zero_division=0)
	conf_matrix = confusion_matrix(y, y_pred, labels=config.targets)

	perm_imp_dict = {label: 0 for label in labels} if labels else {}
	if labels:
		r = permutation_importance(clf, X, y, random_state=42)

		for i in r.importances_mean.argsort()[::-1]:

			perm_imp_dict[labels[i]] = float(r.importances_mean[i])

			# if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
			#print(f"{labels[i]:<8}"
			#	f" {r.importances_mean[i]:.3f}"
			#	f" +/- {r.importances_std[i]:.3f}")

		#print('---')

	if verbose:
		print('Confusion Matrix:')
		print(conf_matrix)
		print()

		for i, y_ in enumerate(y):
			print(f'Next click: {y_}, predicted next click: {y_pred[i]}')

	return macro_accuracy, micro_accuracy, precision, recall, f1, conf_matrix, perm_imp_dict


def evaluate_leave_one_out(clf, X_trains, y_trains, X_tests, y_tests, labels=None, verbose=False, scale=False):
	macro_accuracies = []
	micro_accuracies = []
	precisions = []
	recalls = []
	f1s = []
	conf_matrix_sum = np.array([[0] * len(config.targets)] * len(config.targets))

	perm_imp_lists_dict = {label: [] for label in labels} if labels else {}

	train_len_sum = 0

	for i, X_train in enumerate(X_trains):

		y_train = y_trains[i]
		y_test = y_tests[i]

		if len(X_train) == 0 or len(y_tests[i]) == 0:
			# print('No data!')
			continue

		if scale:
			X_train, X_test = scale_data(X_train, X_tests[i])

		clf = train(clf, X_train, y_train)

		train_len_sum += len(y_train)

		macro_accuracy, micro_accuracy, precision, recall, f1, conf_matrix, perm_imp_dict = evaluate(clf, X_tests[i], y_test, labels=labels, verbose=verbose)

		macro_accuracies.append(macro_accuracy)
		micro_accuracies.append(micro_accuracy)
		precisions.append(precision)
		recalls.append(recall)
		f1s.append(f1)
		conf_matrix_sum += conf_matrix
		if labels:
			for label, perm_imp in perm_imp_dict.items():
				perm_imp_lists_dict[label].append(perm_imp)

	print('Average length of train data:', train_len_sum/len(y_trains))
	print('Average macro accuracy:', sum(macro_accuracies) / len(macro_accuracies))
	print('Average micro accuracy:', sum(micro_accuracies) / len(micro_accuracies))
	print('Average precision:', sum(precisions) / len(precisions))
	print('Average recall:', sum(recalls) / len(recalls))
	print('Average f1:', sum(f1s) / len(f1s))
	print('Confusion matrix (all summed up):')
	print(config.targets)
	print(conf_matrix_sum)
	if len(perm_imp_lists_dict) > 0:
		avg_dict = {}
		print('Average permutation importances:')
		for label, imp_list in perm_imp_lists_dict.items():
			avg_dict[label] = sum(imp_list)/len(imp_list)

		avg_dict_sorted = {k: v for k, v in sorted(avg_dict.items(), key=lambda item: item[1], reverse=True)}
		for label, val in avg_dict_sorted.items():
			print(f'{label}: {val}')

	sum_labels = np.sum(conf_matrix_sum, axis=1)
	sum_labels_percent = sum_labels/sum(sum_labels)
	print()
	print("Distribution of targets:")
	for target, percentage in zip(config.targets, sum_labels_percent):
		print(f'{target}: {round(percentage, 4)}')


def scale_data(X_train, X_test):
	if len(X_train) > 0 and len(X_test) > 0:
		scaler = StandardScaler()
		scaler.fit(X_train)
		return scaler.transform(X_train), scaler.transform(X_test)
	else:
		return X_train, X_test
