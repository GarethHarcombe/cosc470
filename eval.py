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


def merge_intervals(intervals):
    if not intervals:
        return []

    intervals.sort(key=lambda x: x[0])

    merged_intervals = [intervals[0]]

    for i in range(1, len(intervals)):
        current_interval = intervals[i]
        last_merged = merged_intervals[-1]

        if current_interval[0] <= last_merged[1]:
            merged_intervals[-1] = (last_merged[0], max(last_merged[1], current_interval[1]))
        else:
            merged_intervals.append(current_interval)

    return merged_intervals

def intersection_of_intervals(intervals1, intervals2):
    merged_intervals1 = merge_intervals(intervals1)
    merged_intervals2 = merge_intervals(intervals2)

    intersection = []
    
    i = 0
    j = 0

    while i < len(merged_intervals1) and j < len(merged_intervals2):
        interval1 = merged_intervals1[i]
        interval2 = merged_intervals2[j]

        # Check if there is an intersection between the two intervals
        if interval1[1] < interval2[0]:
            i += 1
        elif interval2[1] < interval1[0]:
            j += 1
        else:
            # Calculate the intersection and move forward in both intervals
            intersection_start = max(interval1[0], interval2[0])
            intersection_end = min(interval1[1], interval2[1])
            intersection.append((intersection_start, intersection_end))

            if interval1[1] < interval2[1]:
                i += 1
            else:
                j += 1

    return intersection


def union_of_intervals(intervals1, intervals2):
    merged_intervals1 = merge_intervals(intervals1)
    merged_intervals2 = merge_intervals(intervals2)

    union = []
    
    i = 0
    j = 0

    while i < len(merged_intervals1) and j < len(merged_intervals2):
        interval1 = merged_intervals1[i]
        interval2 = merged_intervals2[j]

        # If interval1 ends before interval2 starts, add interval1 to the union
        if interval1[1] < interval2[0]:
            union.append(interval1)
            i += 1
        # If interval2 ends before interval1 starts, add interval2 to the union
        elif interval2[1] < interval1[0]:
            union.append(interval2)
            j += 1
        else:
            # Calculate the union of overlapping intervals and move forward in both intervals
            union_start = min(interval1[0], interval2[0])
            union_end = max(interval1[1], interval2[1])
            union.append((union_start, union_end))
            
            # Move forward in the interval that ends later
            if interval1[1] < interval2[1]:
                i += 1
            else:
                j += 1

    # Add any remaining intervals from both lists to the union
    union.extend(merged_intervals1[i:])
    union.extend(merged_intervals2[j:])

    return union

def iou(test_labels, predictions):
    sec_overlap = 4

    intervals1 = [(test-sec_overlap/2, test+sec_overlap/2) for test in test_labels]
    intervals2 = [(pred-sec_overlap/2, pred+sec_overlap/2) for pred in predictions]

    intersection = intersection_of_intervals(intervals1, intervals2)
    union = merge_intervals(union_of_intervals(intervals1, intervals2))

    print("Intersection:", intersection)
    print("Union:", union)

    intersection_area = sum([inter[1] - inter[0] for inter in intersection])
    union_area = sum([un[1] - un[0] for un in union])

    return intersection_area / union_area if union_area > 0 else 0


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

    print(iou([1, 2, 3], []))

    print(iou([1, 2, 5], [1, 2]))
