targets = ['brand', 'cart', 'category', 'checkout', 'home', 'product', 'search']
anchors = targets + ['end']

# How many past clicks will be used
clickmemsize = 2

# Data from which experiment(s) should be used (1, 2, 3)
experiment_nums = [1]

# Which models should be tested
baseline = True
distance_model = True
decision_tree = False
random_forest = True
ada_boost = False
gradient_boost = False
logistic_regression = False
