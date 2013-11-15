"""Stat predictor

Usage:
  main.py update <event_config> <state_file> <ts> <value>
  main.py predict <event_config> <state_file> [<ts>]
  main.py get_weights <event_config> <state_file> [<ts>]
  main.py debug <event_config> <state_file>
  main.py train_from_rrd <event_config> <state_file> <rrd_file>
  main.py (-h | --help)

Options:
  -h --help   Show this screen.
"""

from docopt import docopt
import model
import time
import rrd_dump


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


    if arguments['update']:
        state.update(ts, value)
        state.save_state(arguments['<state_file>'])
    elif arguments['predict']:
        print state.predict(ts)
    elif arguments['get_weights']:
        print state.get_prediction_weights(ts)
    elif arguments['train_from_rrd']:
        for ts, value, timewindow in rrd_dump.parse_rrddump_output(rrd_dump.run_rrddump(arguments['<rrd_file>'])):
            state.update(ts, value)

        state.save_state(arguments['<state_file>'])

if __name__ == '__main__':
    arguments = docopt(__doc__, version="Stat predictor 1.0")
    main(arguments)
