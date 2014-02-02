#!/usr/bin/python
import numpy
import yaml
import time
import os.path

class EventType(object):
    def __init__(self, name, duration, num_steps, event_log_filename):
        self.name = name
        self.duration = duration
        self.num_steps = num_steps
        self.periodic = False
        self.event_log_filename = event_log_filename

    def get_timestamps(self):
        # todo use a database or something.
        with open(self.event_log_filename) as event_log:
            for line in event_log:
                event_ts = float(line)
                yield event_ts

    def get_timestamps_in_window(self, begin, end):
        # todo use a database or something.
        for event_ts in self.get_timestamps():
            if begin < event_ts <= end:
                yield event_ts

    def get_prediction_weights(self, ts, slew=None):
        weights = [0] * self.num_steps

        for event_ts in self.get_timestamps_in_window(ts - self.duration, ts):
            step_length = float(self.duration / self.num_steps)
            left = int((ts - event_ts) / step_length)
            right = (left + 1)
            if self.periodic:
                right = right % self.num_steps
            else:
                right = min(right, self.num_steps - 1)

            right_impact = ((ts - event_ts) % step_length) / step_length
            left_impact = 1 - right_impact

            weights[left] += left_impact
            weights[right] += right_impact

        return weights

    def to_dict(self):
        return {
            'name': self.name,
            'duration': self.duration,
            'num_steps': self.num_steps
        }


class PeriodicEvent(EventType):
    # worlds worst OO hierarchy
    def __init__(self, name, duration, num_steps):
        self.name = name
        self.duration = duration
        self.num_steps = num_steps
        self.periodic = True

    def get_timestamps_in_window(self, begin, end):
        first_timestamp = int(begin / self.duration) * self.duration + self.duration
        return xrange(first_timestamp, end, self.duration)

    def to_dict(self):
        return {
            'name': self.name,
            'duration': self.duration,
            'num_steps': self.num_steps,
            'periodic': True,
        }


class StatState(object):
    def __init__(self, event_config, state_file):
        self.load_state(event_config, state_file)

    def load_state(self, event_config_filename, state_filename):
        with open(state_filename) as state_file:
            state_dict = yaml.load(state_file)

        with open(event_config_filename) as event_config:
            event_config_dict = yaml.load(event_config)

        self.events = []
        for event in state_dict['events']:
            if event.get('periodic'):
                self.events.append(PeriodicEvent(event['name'], event['duration'], event['num_steps']))
            else:
                self.events.append(EventType(event['name'], event['duration'], event['num_steps'], event_config_dict[event['name']]))

        self.means = numpy.matrix(state_dict['means']).T
        self.covariance = numpy.matrix(state_dict['covariance'])
        self.measurement_noise = state_dict['measurement_noise']

    def save_state(self, state_filename):
        with open(state_filename, 'w+') as state_file:
            yaml.dump(self.to_dict(), stream=state_file)

    def to_dict(self):
        return {
            'events': [event.to_dict() for event in self.events],
            'means': self.means.T.tolist(),
            'covariance': self.covariance.tolist(),
            'measurement_noise': self.measurement_noise,
        }

    def update(self, ts, measurement):
        print "updating with %s / %s" % (ts, measurement)
        self.means, self.covariance = sorta_kalman(
            self.means,
            self.covariance,
            measurement,
            self.get_prediction_weights(ts),
            self.measurement_noise
        )

    def predict(self, ts):
        C_t = self.get_prediction_weights(ts)

        expected_value = C_t * self.means
        variance = C_t * self.covariance * C_t.T

        return expected_value, variance

    def get_prediction_weights(self, ts):
        weights = []
        for event in self.events:
            weights.extend(event.get_prediction_weights(ts))

        assert len(weights) == len(self.means)
        return numpy.matrix(weights)


def sorta_kalman(means, covariance, measurement, C_t, Q_t):
    # import ipdb; ipdb.set_trace()
    kalman_gain = covariance * C_t.T * (C_t * covariance * C_t.T + Q_t).I
    means = means + kalman_gain * (measurement - C_t * means)

    identity = numpy.identity(means.shape[0])
    covariance = (identity - kalman_gain * C_t) * covariance

    return means, covariance
