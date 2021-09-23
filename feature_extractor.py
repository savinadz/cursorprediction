import os
import math
import random
from datetime import datetime
from collections import Counter

import config
import helpers


def not_same_experiment(event_1, event_2):
    return not event_1 or not event_2 or event_1['experiment_id'] != event_2['experiment_id'] or event_1[
        'session_id'] != event_2['session_id']


class FeatureExtractor:
    """ Includes methods to create train and test data from the given cursor movement data and anchors """

    def __init__(self, use_features=None):
        """ Initializes the Feature vector by providing the relevant features """

        if use_features is None:
            use_features = {'last_clicks': True, 'last_click_dist': True, 'direction': True, 'anchor_dist': True,
                            'speed': True, 'scrolled': True}

        # Last clicked elements
        self.last_clicks = use_features['last_clicks']

        # Distance to the last clicked element
        self.last_click_dist = use_features['last_click_dist']

        # Direction
        self.direction = use_features['direction']

        # Distance to the anchor points
        self.anchor_dist = use_features['anchor_dist']

        # Speed (Pixel/Ms)
        self.speed = use_features['speed']

        # Scrolled
        self.scrolled = use_features['scrolled']

        self.label_vector = []
        if self.last_clicks:
            for i in range(config.clickmemsize):
                for j in range(len(config.targets)):
                    self.label_vector.append(f'last_clicks_{i}_{j}')

        if self.last_click_dist:
            self.label_vector.append('last_click_distance')

        if self.direction:
            self.label_vector.append('direction_1')
            self.label_vector.append('direction_2')

        if self.anchor_dist:
            self.label_vector.extend(['anchor_dist_brand',
                                      'anchor_dist_cart',
                                      'anchor_dist_category',
                                      'anchor_dist_checkout',
                                      'anchor_dist_home',
                                      'anchor_dist_product',
                                      'anchor_dist_search',
                                      'end'])

        if self.speed:
            self.label_vector.append('speed')

        if self.scrolled:
            self.label_vector.append('scrolled')

    def create_feature_vector(self, event, last_event, anchors):
        """ Creates a feature vector of one event using the features from initialization. """

        vector = []

        x1 = event['coords'][0]
        y1 = event['coords'][1]

        # last clicked targets, one hot encoded (#anchors * CLICK_MEMORY_SIZE)
        if self.last_clicks:
            for i in range(config.clickmemsize):
                try:
                    encoding = helpers.get_onehot(event['last_click'][i]['hoverID'])[1]
                    for val in encoding:
                        vector.append(val)
                except (IndexError, TypeError):
                    for _ in range(len(config.targets)):
                        vector.append(0)

        # distance to the last clicked target (1)
        if self.last_click_dist:
            if event['last_click'][-1]:
                x2 = event['last_click'][-1]['coords'][0]
                y2 = event['last_click'][-1]['coords'][1]
                vector.append(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))
            else:
                vector.append(0)

        # cursor direction, using the previous event (2)
        if self.direction:
            if not_same_experiment(event, last_event):
                vector.append(0)
                vector.append(0)
            else:
                x2 = last_event['coords'][0]
                y2 = last_event['coords'][1]
                vector.append(x1 - x2)
                vector.append(y1 - y2)

        # distance to the anchors (#anchors)
        if self.anchor_dist:
            for anchor, anchor_pos in anchors.items():
                x2 = anchor_pos[0]
                y2 = anchor_pos[1]
                vector.append(math.sqrt((x1 - x2) ** 2 + (y2 - y1) ** 2))

        # speed (1)
        if self.speed:

            if not_same_experiment(event, last_event):
                vector.append(0)

            else:
                current_dt = datetime.fromtimestamp(event['timestamp'] / 1000)
                last_dt = datetime.fromtimestamp(last_event['timestamp'] / 1000)
                elapsed_seconds = (current_dt - last_dt).total_seconds()
                vector.append(elapsed_seconds)

        # scrolled (1)
        if self.scrolled:
            if event['scrolled']:
                vector.append(1)
            else:
                vector.append(0)

        return vector

    def preprocess_for_training(self, events):
        """ Chooses relevant datapoints from the collected events (they have to be a mousemove/scroll event
        and have a valid next click event) and creates feature vectors and targets """

        data = []
        targets_ = []

        last_event = None
        for event_list in events:
            for event in event_list:

                if event['event_type'] == 'mousemove' \
                        and 'target' in event['hoverClass'] \
                        and event['next_click'] \
                        and event['next_click']['hoverID']:
                    data.append(
                        self.create_feature_vector(event, last_event, helpers.load_anchors(event['session_id'])))
                    targets_.append(helpers.get_onehot(event['next_click']['hoverID'])[0])

                last_event = event

        return data, targets_

    def load_experiments_for_training(self):
        """ Loads all experiments and prepares for training """

        total_events = []
        for session_id in os.listdir('data/cursordata'):

            if not os.path.isfile(f'anchors/anchors_{session_id}.json'):
                # if they are not already stored, they will be created
                helpers.save_anchors(session_id)

            session_events = []

            for experiment_num in config.experiment_nums:

                if os.path.isdir(f'data/cursordata/{session_id}/{experiment_num}'):

                    for experiment_id in os.listdir(f'data/cursordata/{session_id}/{experiment_num}'):
                        events = helpers.load_experiment(session_id, experiment_num, experiment_id)

                        current_clicks = []
                        current_scroll = False
                        for i, event in enumerate(events):

                            # if last event was scroll event
                            if current_scroll:
                                # store in event that it has been scrolled before
                                event['scrolled'] = True
                            else:
                                # store in event that it hasn't been scrolled before
                                event['scrolled'] = False

                            # if event is scroll event
                            if event['event_type'] == 'scroll':
                                # set scroll flag
                                current_scroll = True
                            # if event is not scroll event
                            else:
                                current_scroll = False

                            event['session_id'] = session_id
                            event['experiment_id'] = experiment_id

                            if event['event_type'] == 'click':
                                # add to list of click events
                                current_clicks.append(event)
                                # if too many values, only keep the last CLICK_MEMORY_SIZE events
                                if len(current_clicks) > config.clickmemsize:
                                    current_clicks = current_clicks[-config.clickmemsize:]

                            else:
                                last_clicks = current_clicks.copy()
                                last_clicks.reverse()
                                # pad with None values
                                while len(last_clicks) < config.clickmemsize:
                                    last_clicks.append(None)

                                last_clicks.reverse()
                                events[i]['last_click'] = last_clicks

                        # add target data (next clicked)
                        events.reverse()
                        current_click = None
                        for i, event in enumerate(events):
                            if event['event_type'] == 'click':
                                current_click = event

                            else:
                                events[i]['next_click'] = current_click

                        events.reverse()
                        session_events.append(events)

                else:
                    print(f"No experiment found: SessionID {session_id} and Experiment num {experiment_num}.")

            total_events.append(session_events)

        random.seed(42)
        random.shuffle(total_events)

        # leave one out evaluation
        X_trains, y_trains, X_tests, y_tests = [], [], [], []

        for i, sess_events in enumerate(total_events):
            # session_events is a list of experiments of the same session
            total_events_copy = total_events.copy()

            test_experiments = sess_events

            # test_experiments = [item for sublist in sess_events for item in sublist]

            total_events_copy.pop(i)
            train_experiments = [item for sublist in total_events_copy for item in sublist]

            """
            # this is just a check.
            test_sess_ids = set()
            for event in [item for sublist in test_experiments for item in sublist]:
                test_sess_ids.add(event['session_id'])

            train_sess_ids = set()
            for event in [item for sublist in train_experiments for item in sublist]:
                train_sess_ids.add(event['session_id'])

            print(test_sess_ids)
            print(train_sess_ids)
            print(test_sess_ids.intersection(train_sess_ids))
            """

            X_train, y_train = self.preprocess_for_training(train_experiments)
            X_test, y_test = self.preprocess_for_training(test_experiments)

            X_trains.append(X_train)
            y_trains.append(y_train)

            X_tests.append(X_test)
            y_tests.append(y_test)


        return X_trains, y_trains, X_tests, y_tests
