#!/usr/bin/python
"""Stat predictor

Usage:
  main.py update <event_config> <state_file> <ts> <value>
  main.py predict <event_config> <state_file> [<ts>]
  main.py predict_many <event_config> <state_file> <start> <length> <step>
  main.py graph_predict_many <event_config> <state_file> <start> <length> <step>
  main.py get_weights <event_config> <state_file> [<ts>] [<slew>]
  main.py get_weights_many <event_config> <state_file> <start> <length> <step> [<slew>]
  main.py debug <event_config> <state_file>
  main.py train_from_rrd <event_config> <state_file> <rrd_file>
  main.py (-h | --help)

Options:
  -h --help   Show this screen.
"""

from docopt import docopt
import numpy
import model
import time
import rrd_dump
import graph


def main(arguments):
    state = model.StatState(
        arguments['<event_config>'],
        arguments['<state_file>'],
    )


    if arguments['<ts>'] is not None:
        ts = float(arguments['<ts>'])
    else:
        ts = time.time()

    if arguments['<value>'] is not None:
        value = float(arguments['<value>'])
    else:
        value = None

    if arguments['<slew>'] is not None:
        slew = float(arguments['<slew>'])
    else:
        slew = None

    if arguments['update']:
        state.update(ts, value)
        state.save_state(arguments['<state_file>'])
    elif arguments['predict']:
        print state.predict(ts)
    elif arguments['predict_many']:
        start = float(arguments['<start>'])
        end = start + float(arguments['<length>'])
        step = float(arguments['<step>'])
        for ts in xrange(start, end, step):
            print state.predict(ts)
    elif arguments['graph_predict_many']:
        start = float(arguments['<start>'])
        end = start + float(arguments['<length>'])
        step = float(arguments['<step>'])

        timestamps = []
        means = []
        errors = []
        for ts in xrange(start, end, step):
            timestamps.append(ts)
            mean, error = state.predict(ts)
            mean = mean.tolist()[0][0]
            error = error.tolist()[0][0]

            means.append(mean)
            errors.append(error)

        graph.graph_predictions(timestamps, means, errors)


    elif arguments['get_weights']:
        print state.get_prediction_weights(ts, slew=slew)
    elif arguments['get_weights_many']:
        start = float(arguments['<start>'])
        end = start + float(arguments['<length>'])
        step = float(arguments['<step>'])

        for ts in numpy.arange(start, end, step):
            print ' '.join([("%0.2f " % weight) for weight in state.get_prediction_weights(ts, slew=slew).tolist()[0] ])

    elif arguments['train_from_rrd']:
        for ts, value, timewindow in rrd_dump.parse_rrddump_output(rrd_dump.run_rrddump(arguments['<rrd_file>'])):
            state.update(ts, value, slew=timewindow)

        state.save_state(arguments['<state_file>'])

if __name__ == '__main__':
    arguments = docopt(__doc__, version="Stat predictor 1.0")
    main(arguments)
