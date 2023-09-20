from dataloader import read_activity
import pandas as pd

from sklearn.model_selection import train_test_split
from models import SlidingWindow, Acceleration


def load_data():
    activities = pd.read_csv("data/activities.csv")[["Activity ID", "Activity Date", "Activity Name", "IS_TRACK_WORKOUT", 
                                                    "HAS_ACCURATE_LAPS", "Activity Type", "Activity Description", "Elapsed Time", 
                                                    "Distance", "Max Heart Rate", "Filename", "Elapsed Time.1", "Moving Time",
                                                    "Distance.1", "Max Speed", "Average Speed", "Elevation Gain"]]
    activities = activities.assign(New_Filename=lambda x: x.Filename.str[:-3])
    return activities[activities.HAS_ACCURATE_LAPS == 1.0]


if __name__ == "__main__":
    activities = load_data()

    train_activities, test_activities = train_test_split(activities, test_size=0.5, random_state=109)

    # train_activities = train_activities[:1]
    # test_activities = test_activities[:1]

    # model = SlidingWindow()
    model = Acceleration()

    if model.requires_training:
        model.train(train_activities)

    errors = model.test(test_activities)

    print(errors)
    print("Average error: ", sum(errors) / len(errors))
    print("done")


    # pred = predictor_acceleration(points)
    # print("Predicted:", pred)
    # print("Actual:   ", laps.time_after_start.values)

    # 