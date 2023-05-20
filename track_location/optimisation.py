from .track_tools import transform_points, add_track_cols
from scipy.optimize import minimize
import numpy as np
import pandas as pd


def estimate_track_location(points):
    """
    estimate_track_location: returns an estimated track location
    Done by taking the top 20 fastest GPS points and taking the mid point

    Inputs ~
        points: pd.DataFrame - GPS points from workout

    Outputs ~
        (float, float) - estimated track location
    """
    fastest_points = points.nlargest(20, "speed")
    return (fastest_points.x.min() + (fastest_points.x.max() - fastest_points.x.min()) / 2, 
            fastest_points.y.min() + (fastest_points.y.max() - fastest_points.y.min()) / 2)


def track_location(points, init=None):
    """
    track_location: finds the location and orientation of the track using optimisation

    Inputs ~
        points: pd.DataFrame - df of GPS points from running workout
        init: (optional) list or array - array of initial parameters (in order): [x_ordinate, y_ordinate, theta]

    Outputs ~
        results of optimisation, use res.x to get parameters
    """

    def track_error(params):
        """
        track_error: get a track location and orientation, calculate the error in the track location
        Error is defined as the negative of the number of points that are within 3 meters of their corresponding projected point

        Inputs ~
            params: list or array - array of initial parameters (in order): [x_ordinate, y_ordinate, theta]

        Outputs ~
            error: int
        """
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

    # filter by points that are sufficiently close to the initial guess and have a relatively high speed
    # improves performance
    points = points[points.is_close & (points.speed > avg_speed)].reset_index(drop=True)

    res = minimize(track_error, init, method="Nelder-Mead")

    return res
