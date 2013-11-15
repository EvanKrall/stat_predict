"""Stat predictor

Usage:
  main.py update <event_config> <state_file> <ts> <value>
  main.py predict <event_config> <state_file> [<ts>]
  main.py get_weights <event_config> <state_file> [<ts>]
  main.py debug <event_config> <state_file>
  main.py (-h | --help)

Options:
  -h --help   Show this screen.
"""

from docopt import docopt
import model
import time


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
        import yaml
        print "before"
        print yaml.dump(state.to_dict())

        state.update(ts, value)
        print "\nafter"
        print yaml.dump(state.to_dict())
    elif arguments['predict']:
        print state.predict(ts)
    elif arguments['get_weights']:
        print state.get_prediction_weights(ts)
    elif arguments['debug']:
        import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    arguments = docopt(__doc__, version="Stat predictor 1.0")
    main(arguments)
