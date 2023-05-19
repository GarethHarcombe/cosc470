from cosc428_track_location.track import Track
from cosc428_track_location.tools import rotate_points
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def estimate_track_location(points):
    fastest_points = points.nlargest(20, "speed")
    return (fastest_points.x.min() + (fastest_points.x.max() - fastest_points.x.min()) / 2, 
            fastest_points.y.min() + (fastest_points.y.max() - fastest_points.y.min()) / 2)


def transform_points(points, x, y, theta, x_col="x", y_col="y"):
    translated_points = (
        points
        .assign(x=lambda df: df[x_col] - x)
        .assign(y=lambda df: df[y_col] - y)
    )

    return rotate_points(translated_points, theta, x_col, y_col)


def add_track_cols(points):
    track = Track()
        
    projected_points = pd.DataFrame(index=range(len(points.x)), columns=["x", "y"])
    for i in range(len(projected_points)):
        projected_points.iloc[i] = track.project(points.x[i], points.y[i])

    # TODO: make this more efficient, can calculate projected points in one go
    new_points = (
        points
        .assign(track_x=projected_points.x)
        .assign(track_y=projected_points.y)
        .assign(dist_to_track=lambda df: ((df.track_x - df.x) ** 2 + (df.track_y - df.y) ** 2) ** 0.5)
    )
    return new_points



def track_location(points, init=None):

    def track_error(params):
        x, y, theta = params
        new_points = transform_points(points, x, y, theta)

        error_calcs = add_track_cols(new_points).assign(error=lambda df: (df.dist_to_track < 3))

        return -error_calcs.error.sum()
    

    if init is None:
        estimate = estimate_track_location(points)
        init = np.array([estimate[0], estimate[1], 0])

    avg_speed = points.speed.mean()

    points = (
        points
        .assign(is_close=lambda df: ((df.x - init[0]) ** 2 + (df.y - init[1]) ** 2) ** 0.5 < 150)
    )

    points = points[points.is_close & (points.speed > avg_speed)].reset_index(drop=True)

    res = minimize(track_error, init, method="Nelder-Mead")

    return res
