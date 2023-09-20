import numpy as np

# Penalty for missing a lap time, error unit in seconds
PENALTY = 100

def evaluate(test_labels, predictions):
    """
    evaluate: given gold standard labels and a model's predictions, return an error for the predictions
    This is calculated as the L1 distance between predicted times and the closest gold standard lap time
    Any gold standard lap times that do not a closest prediction accumulate an additional error given by PENALTY

    Inputs:
        test_labels: list or array (float) - of gold standard labels
        predictions: list or array (float) - model's predictions

    Outputs:
        error: float - error between gold standard labels and predictions
    """
    lap_targeted = [0] * len(test_labels)

    error = 0

    for pred in predictions:
        closest_lap = np.argmin([np.abs(label - pred) for label in test_labels])

        error += np.abs(test_labels[closest_lap] - pred)

        lap_targeted[closest_lap] = 1

    missed_laps = len(test_labels) - sum(lap_targeted)
    error += PENALTY * missed_laps

    # want error in terms of how many gold standard lap times there are
    return error / len(test_labels)


if __name__ == "__main__":
    # test cases:
    # should return 0
    print(evaluate([1, 2, 3], [1, 2, 3]))

    # should return (0.1 + 0.1 + 0.3 + PENALTY * 1) / 3
    print(evaluate([1, 2, 3], [1.1, 1.9, 2.3]))

    # should return (0.5 + 0.5) / 3
    print(evaluate([1, 2, 3], [1, 1.5, 2, 2.5, 3]))

    # should return (0.1 + 0.4 + 0.1) / 3
    print(evaluate([1, 2, 3], [0.9, 1.6, 2.9]))