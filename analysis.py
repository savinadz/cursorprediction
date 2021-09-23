import os
import helpers
import config
import pickle
import pandas as pd
from collections import Counter
from datetime import datetime

df = pickle.load(open('products.pickle', 'rb'))
df = df[df['full_image'] != 'Error:Status:404']


def analyze(experiment_nums, verbose=False, show_scroll=False):

	clicked_targets = []
	total_experiments = 0

	for session_id in os.listdir('data/cursordata'):

		for experiment_num in experiment_nums:

			if os.path.isdir(f'data/cursordata/{session_id}/{experiment_num}'):

				for experiment_id in os.listdir(f'data/cursordata/{session_id}/{experiment_num}'):
					events = helpers.load_experiment(session_id, experiment_num, experiment_id)

					total_experiments += 1

					click_sequence = []

					for i, event in enumerate(events):

						event['session_id'] = session_id

						if show_scroll:
							if event['event_type'] == 'scroll':
									click_sequence.append({'target': 'SCROLL', 'name': '-'})

						if event['event_type'] == 'click':

							for target in config.targets:
								if target in event['hoverID']:
									name = ''

									if target == 'product':
										for _, row in df.iterrows():
											if row['sku'] in event['hoverID']:
												name = row['name']

									event_date = datetime.fromtimestamp(event['timestamp']/1000)

									clicked_targets.append(target)
									click_sequence.append({'target': target, 'name': name, 'date': event_date})
									break

					if verbose:
						print(f'----------SessionID: {session_id}, ExpID: {experiment_id}----------')
						for click_event in click_sequence:
							print(str(click_event['date'])[:-3:] + '\t', click_event['target'], click_event['name'])

						print()

			else:
				if verbose:
					print(f"No experiment found: SessionID {session_id} and Experiment num {experiment_num}.")

	print(Counter(clicked_targets))
	most_common = Counter(clicked_targets).most_common(1)[0][0]
	if verbose:
		print()
		print('----------General----------')
		print('Total click events:', len(clicked_targets))
		print(f'Total experiments (Experiments {experiment_nums}):', total_experiments)
		print('Average num of click events per experiment:', len(clicked_targets)/total_experiments)
		print('Most common target:', most_common)

	return most_common
