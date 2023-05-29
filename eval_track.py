import pandas as pd
import numpy as np
from track_location.optimisation import track_location
from dataloader import read_activity, to_xy


def angle_between_points(x1, y1, x2, y2):
    return np.arctan2(y2 - y1, x2 - x1)


def get_known_tracks(csv_path):
    track_locations = pd.read_csv(csv_path)
    tracks = {}
    for i, row in track_locations[~pd.isna(track_locations.centre_coord1)].iterrows():
        xy = to_xy([row.centre_coord1, row.axis_coord1], 
                   [row.centre_coord2, row.axis_coord2])
        theta = angle_between_points(xy.iloc[0].x, xy.iloc[0].y, xy.iloc[1].x, xy.iloc[1].y) - np.pi / 2
        if theta < 0:
            theta += np.pi
        tracks[row.track_name] = (xy.iloc[0].x, xy.iloc[0].y, theta)

    return tracks


def eval_track_location(model, activities, tracks):
    error = 0

    for i, row in activities.iterrows():
        points, laps, events = read_activity("data/" + row.Filename[:-3], center_points=False)
        pred = model(points)
        print(pred - tracks[row.TRACK])
        example_error = sum([((pred[i] - tracks[row.TRACK][i]) ** 2) ** 0.5 for i in range(len(pred))])
        error += example_error

    return error / len(activities)


def optimisation_method(points):
    # error of 31.275
    pred = track_location(points).x
    pred[2] = pred[2]
    return pred


# from track_location.keypoint_detection_track import get_model, ClassDataset
# import torch
# model = get_model(num_keypoints=2, weights_path="keypointsrcnn_weights.pth")
# dataset_train = ClassDataset(length=16)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# def cv_method(points):
#     images = dataset_train.image_tensor_from_points(points)
#     images = list(image.to(device) for image in images)

#     with torch.no_grad():
#         model.to(device)
#         model.eval()
#         output = model(images)

#     scores = output[0]['scores'].detach().cpu().numpy()

#     high_scores_idxs = [np.argmax(scores)]
#     post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

#     keypoints = []
#     for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
#         keypoints.append([list(map(int, kp[:2])) for kp in kps])

    # return (keypoints[0][0][0], keypoints[0][1][0], 
    #         angle_between_points(keypoints[0][0][0], keypoints[0][1][0], keypoints[0][0][1], keypoints[0][1][1]))


def main():
    track_csv_path = "data/known_track_locations.csv"
    tracks = get_known_tracks(track_csv_path)
    print(tracks)

    activities = (
        pd.read_csv("data/activities.csv")
        [lambda df: ~pd.isna(df.TRACK)]
        [lambda df: df.TRACK.str.contains('|'.join(tracks.keys()))]
    )
    
    print(eval_track_location(optimisation_method, activities, tracks))


if __name__ == "__main__":
    main()
