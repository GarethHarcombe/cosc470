from dataloader import read_activity
import pandas as pd

from sklearn.model_selection import train_test_split
from models import SlidingWindow, Acceleration
from eval_track import get_known_tracks


def load_data(FULL_PIPELINE):
    activities = pd.read_csv("data/activities.csv")[["Activity ID", "Activity Date", "Activity Name", "IS_TRACK_WORKOUT", 
                                                    "HAS_ACCURATE_LAPS", "Activity Type", "Activity Description", "Elapsed Time", 
                                                    "Distance", "Max Heart Rate", "Filename", "Elapsed Time.1", "Moving Time",
                                                    "Distance.1", "Max Speed", "Average Speed", "Elevation Gain", "TRACK"]]

    if FULL_PIPELINE:
        activities = (
            activities
            .assign(New_Filename=lambda x: x.Filename.str[:-3])
            [activities.HAS_ACCURATE_LAPS == 1.0]
        )
    else:
        activities = (
            activities
            .assign(New_Filename=lambda x: x.Filename.str[:-3])
            [activities.HAS_ACCURATE_LAPS == 1.0]
        )
    return activities



def main(DIVISOR=4):
    FULL_PIPELINE = True
    DIVISOR = 4
    activities = load_data(FULL_PIPELINE)

    track_csv_path = "data/known_track_locations.csv"
    tracks = get_known_tracks(track_csv_path)

    train_activities, test_activities = train_test_split(activities, test_size=0.5, random_state=109)

    # train_activities = (
    #     train_activities
    #     [lambda df: ~pd.isna(df.TRACK)]
    #     [lambda df: df.TRACK.str.contains('|'.join(tracks.keys()))]
    # )

    # test_activities = (
    #     test_activities
    #     [lambda df: ~pd.isna(df.TRACK)]
    #     [lambda df: df.TRACK.str.contains('|'.join(tracks.keys()))]
    # )

    # train_activities = train_activities[:1]
    # test_activities = test_activities[:1]

    # model = SlidingWindow(DIVISOR=DIVISOR, FULL_PIPELINE=FULL_PIPELINE)
    model = Acceleration(FULL_PIPELINE=FULL_PIPELINE)

    if model.requires_training:
        model.train(train_activities)

    errors = model.test(test_activities)

    print(errors)
    # print("Average 'precision': ", sum(errors[0]) / len(errors[0]))
    # print("Average 'recall':    ", sum(errors[1]) / len(errors[1]))
    print("Error:", sum(errors) / len(errors))
    print("done")


if __name__ == "__main__":
    main(DIVISOR=3)