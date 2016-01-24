import math

# let's say these are the output of our logistic function
scores = [3.0, 1.0, 0.3]

def softmax(scores):
    """INPUT: a list of scores that were output of a logistic function
    OUTPUT: a list of probabilities for each of those scores, respectively"""

    e = math.e

    # the denominator of each probability is the sum
    # of e to the power of each score
    denom = sum([e**score for score in scores])

    probs = []
    for score in score:
        probs.append((e**score)/denom)

    return probs


print softmax(scores)