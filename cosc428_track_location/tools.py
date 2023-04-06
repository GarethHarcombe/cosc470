
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt


def plot_map(points, x, y, figsize, ax):
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
    MAP_SIZE = 2000
    points.plot.scatter(x=x, 
                        y=y,
                        c="blue",
                        s=1,
                        alpha=0.15,
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


def consistent_scale_plot(points, x="x", y="y", reference_points=None, connected=False):
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

                plot_map(points[i*num_cols + j], x, y, (18, 18 * num_cols ** 2 / num_rows), axis[i, j])
                
                if reference_points is not None:
                    plot_reference(reference_points[i*num_cols + j], x, y, axis[i, j])

        return figure

    else:
        points, reference_points = normalise_to_mean(points, reference_points)
        
        figure, ax = plt.subplots()
        plot_map(points, x, y, (10, 10), ax)

        if reference_points is not None:
            plot_reference(reference_points, x, y, ax)

        if connected:
            points.plot(x, y, ax=ax)

        return figure

