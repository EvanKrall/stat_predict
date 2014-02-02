import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot

def graph_predictions(timestamps, means, errors):
    # print timestamps
    # print means
    # print errors
    
    pyplot.errorbar(timestamps, means, yerr=errors, fmt='ro')

