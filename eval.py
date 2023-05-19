import numpy as np

PENALTY = 5

def evaluate(test_labels, predictions):
    lap_targeted = [0] * len(test_labels)

    error = 0

    for pred in predictions:
        closest_lap = np.argmin([np.abs(label - pred) for label in test_labels])

        error += (test_labels[closest_lap] - pred) ** 2

        lap_targeted[closest_lap] = 1

    missed_laps = len(test_labels) - sum(lap_targeted)
    error += PENALTY * missed_laps

    return error


if __name__ == "__main__":
    # should return 0
    print(evaluate([1, 2, 3], [1, 2, 3]))

    # should return 0.1**2 + 0.1**2 + 0.3**2 + PENALTY * 1
    print(evaluate([1, 2, 3], [1.1, 1.9, 2.3]))

    # should return 0.25 + 0.25
    print(evaluate([1, 2, 3], [1, 1.5, 2, 2.5, 3]))

    # should return 0.01 + 0.16 + 0.01
    print(evaluate([1, 2, 3], [0.9, 1.6, 2.9]))