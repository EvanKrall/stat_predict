#!/usr/bin/python
import numpy
import yaml

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
        step_length = float(self.duration / self.num_steps)

        for event_ts in self.get_timestamps_in_window(ts - self.duration - step_length, ts + step_length):
            location = (ts - event_ts) / step_length
            left = int(numpy.floor(location))
            right = int(numpy.ceil(location))

            right_impact = location - left
            left_impact = 1 - right_impact

            if 0 <= left and left < self.num_steps:
                weights[left] += left_impact
            if 0 <= right and right < self.num_steps:
                weights[right] += right_impact

        if slew:
            num_steps_to_convolve = slew / step_length
            normalizer = 1.0 / num_steps_to_convolve
            convolution = ([normalizer] * int(num_steps_to_convolve)) + [num_steps_to_convolve % 1.0]
            convolved_weights = numpy.convolve(weights, convolution).tolist()

            weights = convolved_weights[:self.num_steps]
            if self.periodic:
                # Periodic things really need to be circularly convolved, so do that.
                weights = [0] * self.num_steps
                convolved_weights.extend([0] * self.num_steps)  # extend so this next loop doesn't reach over the end.

                while len(convolved_weights) > self.num_steps:
                    weights = numpy.add(weights, convolved_weights[:self.num_steps])
                    convolved_weights = convolved_weights[self.num_steps:]

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
        first_timestamp = int(begin / self.duration) * self.duration
        if first_timestamp < begin:
            first_timestamp += self.duration
        assert begin <= first_timestamp
        assert first_timestamp < end
        return numpy.arange(first_timestamp, end, self.duration)

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

    def update(self, ts, measurement, slew=None):
        self.means, self.covariance = sorta_kalman(
            self.means,
            self.covariance,
            measurement,
            self.get_prediction_weights(ts, slew=slew),
            self.measurement_noise
        )

    def predict(self, ts):
        C_t = self.get_prediction_weights(ts)

        expected_value = C_t * self.means
        variance = C_t * self.covariance * C_t.T

        return expected_value, variance

    def get_prediction_weights(self, ts, slew=None):
        weights = []
        for event in self.events:
            weights.extend(event.get_prediction_weights(ts, slew=slew))

        assert len(weights) == len(self.means)
        return numpy.matrix(weights)


def sorta_kalman(means, covariance, measurement, C_t, Q_t):
    # import ipdb; ipdb.set_trace()
    kalman_gain = covariance * C_t.T * (C_t * covariance * C_t.T + Q_t).I
    means = means + kalman_gain * (measurement - C_t * means)

    identity = numpy.identity(means.shape[0])
    covariance = (identity - kalman_gain * C_t) * covariance

    return means, covariance
