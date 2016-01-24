"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np

# SOFTMAX FUNCTIONS
def softmax_my_solution(x):
    """My janky solution to the softmax problem."""
    x = np.array(x)

    if len(x.shape) == 1:
        x.shape = x.shape[0], 1

    probabilities = np.ones_like(x)

    for i in xrange(len(x[0])):
        denom = sum([np.e**num for num in x[:,i]])

        for j in xrange(len(x[:,i])):
            probabilities[j,i] = (np.e**x[j,i]) / denom

    if x.shape[1] == 1:
        probabilities.shape = 1, probabilities.shape[0]
        probabilities = probabilities[0]

    return probabilities

print softmax_my_solution(scores)


def softmax_answer(x):
    """The video's answer to the softmax function."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)

print softmax_answer(scores)


# PLOTTING SOFTMAX CURVES
import matplotlib.pyplot as plt
x = np.linspace(-2.0, 6.0, num=81)  # increases by 0.1, more consistent than np.arange()
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax_answer(scores).T, linewidth=2)


# PLOT BAR CHART SUMS OF PROBABILITIES
x_scores = softmax_answer(scores).T[:,0]
one_scores = softmax_answer(scores).T[:,1]
point_two_scores = softmax_answer(scores).T[:,2]

width = 0.1

p1 = plt.bar(x, x_scores, width, color='b', alpha=0.25)
p2 = plt.bar(x, one_scores, width, color='r', bottom=x_scores, alpha=0.25)
p2 = plt.bar(x, point_two_scores, width, color='g', bottom=x_scores+one_scores, alpha=0.25)

plt.show()