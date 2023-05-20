
import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
from .track import Track


def rotate_points(df, theta, x_col="x", y_col="y"):
    """
    rotate_points: rotates all points around the origin by theta

    Inputs ~
        df: pd.DataFrame of points to be rotated
        theta: float of angle to rotate points

    Outputs ~
        df: pd.DataFrame of rotated points
    """
    # https://academo.org/demos/rotation-about-point/ 

    c, s = np.cos(theta), np.sin(theta)
    j = np.array([[c, s], [-s, c]])
    m = np.dot(j, [df[x_col], df[y_col]])

    df[x_col] = m[0]
    df[y_col] = m[1]
    return df

def transform_points(points, x, y, theta, x_col="x", y_col="y"):
    """
    transform_points: given a set a GPS points, center them around x and y and rotate by theta radians

    Inputs ~
        points: pd.DataFrame - df of GPS points to be transformed
        x: float - x ordinate to center around
        y: float - y ordinate to center around
        theta: float - angle in radians to rotate by
        x_col: (optional) string - df column name of x coordinates
        y_col: (optional) string - df column name of y coordinates

    Outputs ~
        pd.DataFrame - centered and rotated points
    """
    translated_points = (
        points
        .assign(x=lambda df: df[x_col] - x)
        .assign(y=lambda df: df[y_col] - y)
    )

    return rotate_points(translated_points, theta, x_col, y_col)


def add_track_cols(points):
    """
    add_track_cols: add 'track_x', 'track_y', and 'dist_to_track' columns to the given df
    'track_x' and 'track_y' are the respective projected coordinates onto a track
    'dist_to_track' is the L2 distance between the original and projected point

    Input ~
        points: pd.DataFrame - df to add the columns to 

    Output ~
        pd.DataFrame with added columns
    """
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


def plot_map(points, x, y, figsize, ax, MAP_SIZE=2000):
    """
    plot_map: plot small blue dots for each point in the GPS data

    Inputs ~
        points: pd.DataFrame of x, y coordinates to plot
        x: string of column name of x ordinates
        y: string of column name of y ordinates
        figsize: (int, int) of matplotlib plot size
        ax: matplotlib axis to plot points on
    """

    # gives a 4km buffer - should be more than enough to include warm ups. May even decrease later
    points.plot.scatter(x=x, 
                        y=y,
                        c="blue",
                        s=1,
                        # alpha=0.3,
                        xlim=(-MAP_SIZE, MAP_SIZE),
                        ylim=(-MAP_SIZE, MAP_SIZE),
                        figsize=figsize,  # width, height
                        ax=ax)


def plot_reference(points, x, y, ax):
    """
    plot_reference: plot bigger red reference dots on an existing axis

    Inputs ~
        points: pd.DataFrame of x, y coordinates to plot
        x: string of column name of x ordinates
        y: string of column name of y ordinates
        ax: matplotlib axis to plot points on

    Outputs ~
    """
    points.plot.scatter(x=x,
                        y=y,
                        s=3,
                        alpha=0.7,
                        c="red",
                        ax=ax)


def normalise_to_mean(points, reference_points=None):
    """
    normalise_to_mean: center points around the mean of points

    Inputs ~
        points: pd.DataFrame of x, y coordinates to plot
        reference_points: optional pd.DataFrame of x, y reference points to plot

    Outputs ~ 
        (points, reference_points):
            points: pd.DataFrame with updated coordinates
            reference_points: pd.DataFrame with updated coordinates
    """

    x_mean = points.x.mean()
    y_mean = points.y.mean()

    points = (
        points
        .assign(x=lambda x: x.x - x_mean)
        .assign(y=lambda x: x.y - y_mean)
    )

    if reference_points is not None:
        reference_points = (
            reference_points
            .assign(x=lambda x: x.x - x_mean)
            .assign(y=lambda x: x.y - y_mean)
        )
    
    return (points, reference_points)


def consistent_scale_plot(points, x="x", y="y", reference_points=None, connected=False, MAP_SIZE=2000):
    """
    consistent_scale_plot: Plot points using a consistent scale (defined in plot_map function)

    Inputs ~
        points: pd.DataFrame of x, y coordinates to plot
        x: string of column name of x ordinates
        y: string of column name of y ordinates
        reference_points: optional pd.DataFrame of x, y reference points to plot
        connected: bool, if true then consecutive points will be connected by lines

    Outputs ~
        Returns the matplotlib figure of the map
    """

    if type(points) == list:
        num_rows = 3
        num_cols = 2
        figure, axis = plt.subplots(num_rows, num_cols)
        for i in range(num_rows):
            for j in range(num_cols):
                if reference_points is None:
                    points[i*num_cols + j], reference_points = normalise_to_mean(points[i*num_cols + j], None)
                else:
                    points[i*num_cols + j], reference_points[i*num_cols + j] = normalise_to_mean(points[i*num_cols + j], reference_points[i*num_cols + j])

                plot_map(points[i*num_cols + j], x, y, (18, 18 * num_cols ** 2 / num_rows), axis[i, j], MAP_SIZE=MAP_SIZE)
                
                if reference_points is not None:
                    plot_reference(reference_points[i*num_cols + j], x, y, axis[i, j])

        return figure

    else:
        points, reference_points = normalise_to_mean(points, reference_points)
        
        figure, ax = plt.subplots()
        plot_map(points, x, y, (10, 10), ax, MAP_SIZE=MAP_SIZE)

        if reference_points is not None:
            plot_reference(reference_points, x, y, ax)

        if connected:
            points.plot(x, y, ax=ax)

        return figure

