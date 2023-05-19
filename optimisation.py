from cosc428_track_location.track import Track
from cosc428_track_location.tools import rotate_points
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def estimate_track_location(points):
    fastest_points = points.nlargest(20, "speed")
    return (fastest_points.x.min() + (fastest_points.x.max() - fastest_points.x.min()) / 2, 
            fastest_points.y.min() + (fastest_points.y.max() - fastest_points.y.min()) / 2)


def track_location(points, init=None):

    def track_error(params):
        x, y, theta = params
        track = Track()
        translated_points = (
            points
            .assign(x=lambda df: df.x - x)
            .assign(y=lambda df: df.y - y)
        )

        new_points = rotate_points(translated_points, theta)
        projected_points = pd.DataFrame(index=range(len(new_points.x)), columns=["x", "y"])
        for i in range(len(projected_points)):
            projected_points.iloc[i] = track.project(new_points.x[i], new_points.y[i])

        # TODO: make this more efficient, can calculate projected points in one go
        error_calcs = (
            new_points
            .assign(proj_x=projected_points.x)
            .assign(proj_y=projected_points.y)
            .assign(dist=lambda df: ((df.proj_x - df.x) ** 2 + (df.proj_y - df.y) ** 2) ** 0.5)
            .assign(error=lambda df: (df.dist < 3))
            # .assign(error=lambda df: df.is_close * df.dist)
            # .assign(error=lambda df: df.dist)
        )

        return -error_calcs.error.sum()
        # return error_calcs.error.sum()
    

    if init is None:
        estimate = estimate_track_location(points)
        init = np.array([estimate[0], estimate[1], 0])
        # init = np.array([points.x.mean(), points.y.mean(), 0])

    avg_speed = points.speed.mean()

    points = (
        points
        .assign(is_close=lambda df: ((df.x - init[0]) ** 2 + (df.y - init[1]) ** 2) ** 0.5 < 150)
    )

    points = points[points.is_close & (points.speed > avg_speed)].reset_index(drop=True)

    res = minimize(track_error, init, method="Nelder-Mead")

    return res
