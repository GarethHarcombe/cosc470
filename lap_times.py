from dataloader import read_activity
import pandas as pd

from sklearn.model_selection import train_test_split
from models import SlidingWindow, Acceleration, Encoder
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
    DIVISOR = 7
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
    # model = Acceleration(FULL_PIPELINE=FULL_PIPELINE)
    model = Encoder(FULL_PIPELINE=FULL_PIPELINE)

    if model.requires_training:
        print("Training...")
        for i in range(model.epochs):
            print("Epoch ", i)
            model.train(train_activities)
            # model.save("sliding_window_model.pkl")
            # model.load("sliding_window_model.pkl")

            errors = model.test(test_activities, metric="both")
            print("Average 'precision': ", sum([error[0] for error in errors]) / len(errors))
            print("Average 'recall':    ", sum([error[1] for error in errors]) / len(errors))
            print("Average 'IoU':       ", sum([error[2] for error in errors]) / len(errors))
            # input()

    errors = model.test(test_activities, metric="both")

    print(errors)
    print("Average 'precision': ", sum([error[0] for error in errors]) / len(errors))
    print("Average 'recall':    ", sum([error[1] for error in errors]) / len(errors))
    print("Average 'IoU':       ", sum([error[2] for error in errors]) / len(errors))
    # print("Error:", sum(errors) / len(errors))
    print("done")


if __name__ == "__main__":
    main(DIVISOR=4)