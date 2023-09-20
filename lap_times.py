from dataloader import read_activity
import pandas as pd
import datetime as dt
from eval import evaluate
from sklearn.model_selection import train_test_split
from sklearn import svm
import random


TIME_THRESHOLD = 12
ACCELERATION_THRES = 0.5

def grouping_1d(preds):
    """Given an input series of timestamps, 
    group timestamps together and only take the first timestamp."""
    i = 0
    final_preds = []
    while i < len(preds):
        group = preds[(preds[i] <= preds) & 
                      (preds <= preds[i] + TIME_THRESHOLD)]
        # pick the first one as we want 
        # the start of the lap/change in speed
        # final_preds.append(group.iloc[0])
        final_preds.append(group.mean())
        i = group.last_valid_index() + 1

    return final_preds

def predictor_acceleration(points):
    """Simple predictor that makes predictions off the acceleration of the athlete"""
    points["acceleration"] = points.speed.diff() / points.Timestamp.diff().dt.total_seconds()
    points["abs_acceleration"] = points.acceleration.abs()

    predictions = points[points.abs_acceleration > ACCELERATION_THRES].time_after_start.reset_index(drop=True)
    return grouping_1d(predictions)


def load_data():
    activities = pd.read_csv("data/activities.csv")[["Activity ID", "Activity Date", "Activity Name", "IS_TRACK_WORKOUT", 
                                                    "HAS_ACCURATE_LAPS", "Activity Type", "Activity Description", "Elapsed Time", 
                                                    "Distance", "Max Heart Rate", "Filename", "Elapsed Time.1", "Moving Time",
                                                    "Distance.1", "Max Speed", "Average Speed", "Elevation Gain"]]
    activities = activities.assign(New_Filename=lambda x: x.Filename.str[:-3])
    return activities[activities.HAS_ACCURATE_LAPS == 1.0]


def generate_slices(points, laps, events):
    """df with 6 rows, third row is at t
    1 if there is a lap time in interval [t, t+1)
    0 if not"""
    time_slice_duration = 1  # in seconds
    num_padding_points = 7   # number of points either side of the query time.

    start_time = int(points.iloc[num_padding_points].time_after_start)
    end_time = int(points.iloc[-num_padding_points-1].time_after_start)

    data = []
    labels = []
    times = []

    for t in range(start_time, end_time, time_slice_duration):
        label = 1 if len(laps[(t <= laps.time_after_start) & (laps.time_after_start < t+1)]) > 0 else 0

        query_time = pd.Timestamp(events.iloc[0].Timestamp + dt.timedelta(seconds=t))
        closest_index = points.set_index('Timestamp').index.get_indexer([query_time], method='nearest')[0]

        subset = points[closest_index-num_padding_points:closest_index+num_padding_points][["track_x", "track_y", 
                                                                                            "dist_to_track", "speed",
                                                                                            "acceleration", "time_after_start"]]
        subset["time"] = subset.time_after_start - t
        subset = subset.drop(columns=["time_after_start"])

        data.append(subset.values.ravel())
        labels.append(label)
        times.append(t)

    return data, labels, times


def generate_data(activities):
    data = []
    labels = []
    window_times = []
    all_laps = []

    for file in activities.New_Filename.values:
        print(file)
        # points, laps, events = read_activity("data/" + file)
        # points.to_pickle("data/" + file[:-3] + "points" + file[-4:])
        # laps.to_pickle("data/"   + file[:-3] + "laps"   + file[-4:])
        # events.to_pickle("data/" + file[:-3] + "events" + file[-4:])
        points = pd.read_pickle("data/" + file[:-3] + "points" + file[-4:])
        laps = pd.read_pickle("data/"   + file[:-3] + "laps"   + file[-4:])
        events = pd.read_pickle("data/" + file[:-3] + "events" + file[-4:])
        new_data, new_labels, times = generate_slices(points, laps, events)
        data.append(new_data)
        labels.append(new_labels)
        window_times.append(times)
        all_laps.append(laps)

    return data, labels, window_times, all_laps


def generate_predictions(clf, test_data, test_window_times):
    output = []

    for data, window_times in zip(test_data, test_window_times):
        test_labels_preds = clf.predict(data)
        workout_output = []

        for i, label in enumerate(test_labels_preds):
            if label == 1:
                workout_output.append(window_times[i])

        output.append(workout_output)

    return output



if __name__ == "__main__":
    activities = load_data()

    train_activities, test_activities = train_test_split(activities, test_size=0.5, random_state=109)

    # train_activities = train_activities[:1]
    # test_activities = test_activities[:1]

    train_data, train_labels, _, _ = generate_data(train_activities)

    flattened_data = []
    flattened_labels = []

    for data, label in zip(train_data, train_labels):
        flattened_data += data
        flattened_labels += label

    print("Num training examples", len(flattened_data))

    pos_trains = sum(flattened_labels)  # number of positive training examples
    neg_trains = len(flattened_labels) - pos_trains
    
    neg_indices = [i for i, label in enumerate(flattened_labels) if label == 0]
    pos_indices = [i for i, label in enumerate(flattened_labels) if label == 1]

    sampled_indices = random.sample(neg_indices, pos_trains * 3)
    new_indices = set(pos_indices) | set(sampled_indices)

    condensed_data = []
    condensed_labels = []

    for i, (data, label) in enumerate(zip(flattened_data, flattened_labels)):
        if i in new_indices:
            condensed_data.append(data)
            condensed_labels.append(label)

    print("New Num training examples", len(condensed_data))
    clf = svm.SVC(kernel='linear')#, class_weight="balanced")

    """
    Example with balanced class weights:
    [13, 1152, 1168, 2042, 2055, 2068, 2106, 2122, 2135, 2148, 2161, 2193, 2208, 2221, 2234, 2247, 2279, 2294, 2307, 2320, 2333, 2365, 2382, 2395, 2408, 2421, 2454, 2467, 2480, 2493, 2506, 2541, 2555, 2568, 2581, 2594, 2627, 2652, 2665, 2678, 2709, 2725, 2738, 2751, 2764, 2796, 2809, 2822, 2835, 2848, 2883, 2896, 2909, 2922, 2935, 2967, 2983, 2996, 3009, 3022, 3164, 3177, 3190, 3203, 3216, 3229, 3242, 3255, 3268, 3281, 3294, 3308, 3321, 3334, 3347, 3360, 4258, 4273, 4286, 4299, 4312, 4325]
    [1176.953 1206.814 1268.614 1302.134 1356.12  1387.623 1442.091 1473.462
    1529.818 1561.576 1616.888 1649.853 1705.649 1737.835 1788.852 1820.764
    1874.081 1906.27  1957.78  1990.09  2043.687 2076.099 2129.113 2161.879
    3216.031]
    """

    clf.fit(condensed_data, condensed_labels)

    test_data, test_labels, test_window_times, laps = generate_data(test_activities)

    output = generate_predictions(clf, test_data, test_window_times)
    
    errors = []

    for i, workout_output in enumerate(output):
        workout_output = grouping_1d(pd.Series(workout_output))  # thin the predictions
        print("Predicted: ", workout_output)
        print("Actual: ", laps[i].time_after_start.values)
        errors.append(evaluate(laps[i].time_after_start.values, workout_output))

    print(errors)
    print("Average error: ", sum(errors) / len(errors))
    print("done")

    # points, laps, events = read_activity("data/activities/5431221291.fit")
    # data = generate_slices(points, laps, events)


    # pred = predictor_acceleration(points)
    # print("Predicted:", pred)
    # print("Actual:   ", laps.time_after_start.values)

    # 