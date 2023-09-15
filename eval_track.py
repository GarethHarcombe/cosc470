import pandas as pd
import numpy as np
from track_location.optimisation import track_location
from dataloader import read_activity, to_xy
from track_location.keypoint_detection_track import get_model, ClassDataset
import torchvision
import torch
import track_location.convolution as convolution


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

def angle_diff(angle1, angle2):
    return abs((angle1 - angle2 + np.pi/2) % (np.pi) - np.pi/2)


def eval_track_location(model, activities, tracks):
    errors = []

    for _, row in activities.iterrows():
        points, _, _ = read_activity("data/" + row.Filename[:-3], center_points=False)
        pred = model(points)
        error = (((pred[0] - tracks[row.TRACK][0])**2 + 
                  (pred[1] - tracks[row.TRACK][1])**2)**0.5,
                 angle_diff(pred[2], tracks[row.TRACK][2]))
        errors.append(error)
        print(error)

    return (sum(error[0] for error in errors) / len(errors),
            sum(error[1] for error in errors) / len(errors))


def optimisation_method(points):
    # error: 
    pred = track_location(points).x
    pred[2] = pred[2]
    return pred


class KeypointMethod:
    def __init__(self):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = get_model(num_keypoints=2, weights_path="track_location/keypointsrcnn_weights.pth", device=self.device)
        self.dataset_train = ClassDataset(length=16)
        self.model.to(self.device)
        self.model.eval()
        

    def cv_method(self, points):
        image = self.dataset_train.image_tensor_from_points(points)
        images = image.to(self.device)[None, :]

        with torch.no_grad():
            output = self.model(images)

        scores = output[0]['scores'].detach().cpu().numpy()

        high_scores_idxs = [np.argmax(scores)]
        post_nms_idxs = torchvision.ops.nms(output[0]['boxes'][high_scores_idxs], output[0]['scores'][high_scores_idxs], 0.3).cpu().numpy() # Indexes of boxes left after applying NMS (iou_threshold=0.3)

        keypoints = []
        for kps in output[0]['keypoints'][high_scores_idxs][post_nms_idxs].detach().cpu().numpy():
            keypoints.append([list(kp[:2]) for kp in kps])
        print(keypoints)
        x1 = self.dataset_train.image_coords_to_x_point(keypoints[0][0][0], points.x.mean())
        y1 = self.dataset_train.image_coords_to_y_point(keypoints[0][0][1], points.y.mean())
        x2 = self.dataset_train.image_coords_to_x_point(keypoints[0][1][0], points.x.mean())
        y2 = self.dataset_train.image_coords_to_y_point(keypoints[0][1][1], points.y.mean())
        print(x1, y1, x2, y2)
        result = [x1, y1, angle_between_points(x1, y1, x2, y2)]
        print(result)
        return result



class ConvolutionMethod:
    # error: (4.200736483630741, 0.08025518187695763)
    def __init__(self):
        self.dataset_train = ClassDataset(length=16)
        

    def cv_method(self, points):
        image = self.dataset_train.image_tensor_from_points(points, format="numpy")
        track_location = convolution.predict_track_location(image)
        
        x1 = self.dataset_train.image_coords_to_x_point(track_location[0], points.x.mean())
        y1 = self.dataset_train.image_coords_to_y_point(track_location[1], points.y.mean())
        result = [x1, y1, track_location[2]]
        return result


def main():
    track_csv_path = "data/known_track_locations.csv"
    tracks = get_known_tracks(track_csv_path)
    print(tracks)

    # model = optimisation_method
    # model = KeypointMethod().cv_method
    model = ConvolutionMethod().cv_method

    activities = (
        pd.read_csv("data/activities.csv")
        [lambda df: ~pd.isna(df.TRACK)]
        [lambda df: df.TRACK.str.contains('|'.join(tracks.keys()))]
    )
    

    print(f"Final results: {eval_track_location(model, activities, tracks)}")


if __name__ == "__main__":
    main()
