#!/usr/bin/python
import numpy
import yaml

class ImpulseModel(object):
    """Models an impulse to a type of event, e.g. a deployment, by breaking the period directly after the event into a set number of chunks."""
    def __init__(self, name, duration, num_steps, event_log_filename):
        self.name = name
        self.duration = duration
        self.num_steps = num_steps
        self.periodic = False
        self.event_log_filename = event_log_filename

    def copy(self):
        return ImpulseModel(self.name, self.duration, self.num_steps, self.periodic, self.event_log_filename)

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
        # Todo: steal the rewritten weight/slew code from the PeriodicModel.
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

class PeriodicModel(object):
    """Models a periodic trend, by breaking the period into a pre-defined number of blocks, and linearly interpolating between them."""
    def __init__(self, name, duration, num_steps):
        self.name = name
        self.duration = duration
        self.num_steps = num_steps
        self.periodic = True

    def copy(self):
        return PeriodicModel(self.name, self.duration, self.num_steps)

    def get_prediction_weights(self, ts, slew=None):
        ts = float(ts)
        weights = numpy.zeros(self.num_steps)

        if slew:
            for coeff, time in slew_ts(ts, slew, (self.duration / self.num_steps / 12)):
                weights += coeff * self.get_prediction_weights(time)
            return weights
        else:
            step_length = self.duration / float(self.num_steps)

            weight_right = (ts % step_length) / step_length
            weight_left = 1.0 - weight_right

            left = numpy.floor((ts % self.duration) / self.duration * self.num_steps)
            right = (left+1) % self.num_steps

            weights[left] = weight_left
            weights[right] = weight_right
            return weights

    def to_dict(self):
        return {
            'name': self.name,
            'duration': self.duration,
            'num_steps': self.num_steps,
            'periodic': True,
        }

def slew_ts(ts, slew, resolution):
    if slew < resolution:
        return [(1.0, ts + slew / 2.0)]

    times = numpy.arange(
        ts,
        ts + slew,
        resolution
    )
    return ((1.0/len(times), time) for time in times)


class RingBuffer(object):
    """a convenient way to store the last N things. Should have fast, constant-time access."""
    def __init__(self, size_or_list):
        if hasattr(size_or_list, '__iter__'):
            self.size = len(size_or_list)
            self.buffer = numpy.concatenate([numpy.array(size_or_list, dtype='float64'), numpy.array(size_or_list, dtype='float64')])
            self.i = 0
        else:
            self.size = size_or_list
            self.buffer = numpy.zeros(size_or_list * 2)
            self.i = 0

    def push(self, val):
        self[0] = val
        self.i += 1
        self.i = self.i % self.size

    def __setitem__(self, i, val):
        self.buffer[self.getindex(i)] = self.buffer[self.getindex(i) + self.size] = val

    def getindex(self, i):
        return (i + self.i) % self.size

    def __getitem__(self, i):
        if isinstance(i, slice):
            start = self.getindex(i.start) if (i.start is not None) else self.i
            stop = start + (i.stop - i.start) if (i.stop is not None) else self.i + self.size
            index = slice(start, stop, i.step)
        else:
            index = self.getindex(i)
        return self.buffer[index]

    def __len__(self):
        return self.size

    def __iter__(self):
        return self.buffer[self.i : self.i + self.size].__iter__()


class Arma(object):
    """Auto-regressive moving average model."""
    def __init__(self, p, q):
        self.past_y = RingBuffer(p)
        self.past_epsilon = RingBuffer(q)

    def copy(self):
        return Arma(self.past_y[:], self.past_epsilon[:])

    def report(self, ts, measurement, exogenous, prediction_error, slew):
        # print '%f,%f,%f,%f,%f,%f' % (ts, measurement, exogenous, prediction_error, predicted_value, slew or 0.0)
        if not slew:
            self.past_y.push((measurement - exogenous) or 0.0)
            self.past_epsilon.push(prediction_error or 0.0)

    def get_prediction_weights(self, ts, slew=None):
        if slew:
            # todo is it possible to do slew at all in arma?
            return numpy.zeros(len(self.past_y) + len(self.past_epsilon))
        return numpy.concatenate([self.past_y[:], self.past_epsilon[:]])

    def to_dict(self):
        return {
            'arma': True,
            'past_y': self.past_y[:].tolist(),
            'past_epsilon': self.past_epsilon[:].tolist(),
        }

class Constant(ImpulseModel):
    def __init__(self):
        pass
    def copy(self):
        return self
    def get_prediction_weights(self, ts, slew=None):
        return numpy.array([1])

    def to_dict(self):
        return {
            'constant': True,
        }


class StatState(object):
    @classmethod
    def load_state(cls, event_config_filename, state_filename):
        with open(state_filename) as state_file:
            state_dict = yaml.load(state_file)

        with open(event_config_filename) as event_config:
            event_config_dict = yaml.load(event_config)

        return cls(state_dict, event_config_dict)

    def __init__(self, state_dict, event_config_dict):
        self.ts = state_dict.get('ts', 0)
        self.resolution = state_dict.get('resolution', None)

        self.events = []
        for event in state_dict['events']:
            if event.get('arma'):
                self.events.append(Arma(event['past_y'], event['past_epsilon']))
            elif event.get('constant'):
                self.events.append(Constant())
            elif event.get('periodic'):
                self.events.append(PeriodicModel(event['name'], event['duration'], event['num_steps']))
            else:
                self.events.append(ImpulseModel(event['name'], event['duration'], event['num_steps'], event_config_dict[event['name']]))

        self.means = numpy.matrix(state_dict['means']).T
        self.covariance = numpy.matrix(state_dict['covariance'])
        self.measurement_noise = state_dict['measurement_noise']

        self.variance_alpha = state_dict['variance_alpha']
        self.variance_ewma = state_dict.get('variance_ewma', None)

    def save_state(self, state_filename):
        with open(state_filename, 'w+') as state_file:
            yaml.dump(self.to_dict(), stream=state_file)

    def to_dict(self):
        return {
            'ts': self.ts,
            'resolution': self.resolution,
            'events': [event.to_dict() for event in self.events],
            'means': self.means.T.tolist(),
            'covariance': self.covariance.tolist(),
            'measurement_noise': self.measurement_noise,
            'variance_alpha': self.variance_alpha,
            'variance_ewma': float(self.variance_ewma),
        }

    def update(self, ts, measurement, slew=None):
        if ts < self.ts:
            raise ValueError("You're trying to go back in time! %f < %f" % (ts, self.ts))

        self.resolution = ts - self.ts
        self.ts = ts
        assert slew <= self.resolution

        if not numpy.isnan(measurement):
            self.means, self.covariance, prediction_error = sorta_kalman(
                self.means,
                self.covariance,
                measurement,
                self.get_prediction_weights(ts, slew=slew),
                self.measurement_noise
            )

            self.report(ts, measurement, prediction_error, slew)

            print ','.join('%f' % x for x in ([ts, measurement, prediction_error] + self.means.T.tolist()[0]))
        else:
            self.report(ts, numpy.nan, numpy.nan, slew)

    def report(self, ts, measurement, prediction_error, slew):
        if self.variance_ewma is not None:
            self.variance_ewma = self.variance_ewma * self.variance_alpha + (prediction_error ** 2) * (1.0 - self.variance_alpha)
        else:
            self.variance_ewma = (prediction_error ** 2)

        for event in self.events:
            if hasattr(event, 'report'):
                exogenous, _ = self.predict(ts, ignore=event)
                predicted_value, _ = self.predict(ts)
                event.report(ts, measurement, exogenous, prediction_error, slew)

    def predict(self, ts, slew=None, ignore=None):
        C_t = self.get_prediction_weights(ts, slew=None, ignore=ignore)

        expected_value = C_t * self.means
        variance = C_t * self.covariance * C_t.T

        return expected_value, variance

    def predict_monte_carlo(self, length):
        # sample parameters from means/covariances
        sample = numpy.matrix(self.means.flat)
        # sample = numpy.matrix(numpy.random.multivariate_normal(numpy.array(self.means.flat), self.covariance))

        # create new copy of each event. (man, I really need to rename 'event' to something else)
        events = [event.copy() for event in self.events]

        # simulate
        for ts in numpy.arange(self.ts, self.ts + length, self.resolution):
            weights = numpy.matrix(numpy.concatenate([event.get_prediction_weights(ts) for event in events]))
            predicted_value = weights * sample.T

            yield predicted_value.flat[0]

            epsilon = numpy.sqrt(self.variance_alpha) * numpy.random.normal()

            for event in events:
                if hasattr(event, 'report'):
                    other_weights = numpy.matrix(numpy.concatenate([(0 if (event == e) else 1) * e.get_prediction_weights(ts) for e in events]))
                    exogenous = other_weights * sample.T
                    event.report(
                        ts,
                        predicted_value + epsilon,
                        exogenous,
                        epsilon,
                        0.0,
                    )

    def get_prediction_weights(self, ts, slew=None, ignore=None):
        # apologies for the really long one-liner.
        weights = numpy.concatenate([(0 if (ignore == event) else 1) * event.get_prediction_weights(ts, slew=slew) for event in self.events])

        assert len(weights) == len(self.means)
        return numpy.matrix(weights)


def sorta_kalman(means, covariance, measurement, C_t, Q_t):
    kalman_gain = covariance * C_t.T * (C_t * covariance * C_t.T + Q_t).I
    prediction_error = measurement - C_t * means
    means = means + kalman_gain * (prediction_error)

    identity = numpy.identity(means.shape[0])
    covariance = (identity - kalman_gain * C_t) * covariance

    return means, covariance, prediction_error
