import numpy as np

# Penalty for missing a lap time, error unit in seconds
PENALTY = 500

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


def calculate_errors(numbers, target_numbers):
    """
    Calculate the error between each item in a list of floats and their
    closest corresponding numbers in another list.

    Args:
    numbers (list of float): List of floats for which to calculate errors.
    target_numbers (list of float): List of target numbers for comparison.

    Returns:
    list of float: List of errors, where each error corresponds to the
                   difference between a number in the 'numbers' list and its
                   closest target number.
    """
    if len(numbers) == 0:
        return 0

    errors = []

    for number in numbers:
        closest_target = min(target_numbers, key=lambda x: abs(x - number), default=0)
        error = abs(number - closest_target)
        errors.append(error)

    return sum(errors) / len(errors)


def evaluate(test_labels, predictions):
    # precision metric - measure the quality of predictions, how close they are to ground truths
    precision = calculate_errors(test_labels, predictions)

    # recall metric    - measure the quantity of predictions, how well all the predicted times match the ground truths
    recall = calculate_errors(predictions, test_labels)

    return precision, recall


if __name__ == "__main__":
    ##### OLD TESTS
    # # test cases:
    # # should return 0
    # print(evaluate([1, 2, 3], [1, 2, 3]))

    # # should return (0.1 + 0.1 + 0.3 + PENALTY * 1) / 3
    # print(evaluate([1, 2, 3], [1.1, 1.9, 2.3]))

    # # should return (0.5 + 0.5) / 3
    # print(evaluate([1, 2, 3], [1, 1.5, 2, 2.5, 3]))

    # # should return (0.1 + 0.4 + 0.1) / 3
    # print(evaluate([1, 2, 3], [0.9, 1.6, 2.9]))
    
    print(evaluate([1, 2, 3], [1, 2, 3]))

    print(evaluate([1, 2, 3], [1.1, 1.9, 2.3]))

    print(evaluate([1, 2, 3], [1, 1.5, 2, 2.5, 3]))

    print(evaluate([1, 2, 3], [0.9, 1.6, 2.9]))

    print(evaluate([1, 2, 3], []))