from matplotlib import pyplot as plt

def graph_predictions(timestamps, means, errors):
    plt.errorbar(timestamps, means, yerr=errors, fmt='ro')
    plt.show()

def graph_means_and_covariances(means, covariance):
	fig = plt.figure()

	ax1 = fig.add_subplot(2,2,1)
	ax1.matshow(covariance, cmap=plt.cm.rainbow)

	ax3 = fig.add_subplot(2,2,2)
	ax3.matshow(covariance, cmap=plt.cm.coolwarm)

	ax2 = fig.add_subplot(2,1,2)
	ax2.bar(range(len(means)), means)

	plt.show()