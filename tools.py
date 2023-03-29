# Gareth 06/03/2023 - Data Visualisation and other tools

import numpy as np
import pandas as pd
from matplotlib import cm
import fitdecode
import sweat
import matplotlib.pyplot as plt


MARGIN = 1
R = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
HOME_DIR = "/home/gareth/Documents/Uni/2023/cosc470/"


def haversine(lat1, long1, lat2, long2):
    """
    Converted to be vectorised from:
    https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points 
    
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians 
    if type(long1) == pd.core.series.Series:
        long1, lat1, long2, lat2 = map(np.radians, [long1.astype(float), lat1.astype(float), 
                                                    long2.astype(float), lat2.astype(float)])
    else:
        long1, lat1, long2, lat2 = map(np.radians, [long1, lat1, long2, lat2])

    # haversine formula 
    dlon = long2 - long1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    
    return c * R


def read_activity(activity):
    if activity[-3:] == "fit":
        return read_fit(activity)
    elif activity[-3:] == "tcx":
        return read_tcx(activity)


def read_tcx(activity):
    points = pd.DataFrame(columns=['Timestamp', 'lat', 'long', 'speed', 'cadence'])
    laps = pd.DataFrame(columns=['Timestamp', 'start_pos_lat', 'start_pos_long', 'end_pos_lat',
                                 'end_pos_long', 'total_timer_time', 'total_distance', 'avg_speed'])
    
    data = sweat.read_tcx(HOME_DIR + activity)

    points = data[data.lap == 0]
    laps   = data[data.lap == 1]

    return (points, laps)



def get_frame_field(frame, field):
    """Checks if the frame has a specific field, returns None if not"""
    if frame.has_field(field):
        return frame.get_value(field)
    return None


def read_fit(activity):
    """
    Input: a string directory of an activity file
    Reads all the relevant fields for data points and lap times
    Output: a tuple of datapoints and laps df
    """
    points = pd.DataFrame(columns=['Timestamp', 'lat', 'long', 'speed', 'cadence'])
    laps = pd.DataFrame(columns=['Timestamp', 'start_pos_lat', 'start_pos_long', 'end_pos_lat',
                                 'end_pos_long', 'total_timer_time', 'total_distance', 'avg_speed'])
    
    with fitdecode.FitReader(HOME_DIR + activity) as fit_file:
        for frame in fit_file:         
            if isinstance(frame, fitdecode.records.FitDataMessage):
                
                if frame.name == 'lap':
                    timestamp      = get_frame_field(frame, 'timestamp')
                    start_pos_lat  = get_frame_field(frame, 'start_position_lat')
                    start_pos_long = get_frame_field(frame, 'start_position_long')
                    end_pos_lat    = get_frame_field(frame, 'end_position_lat')
                    end_pos_long   = get_frame_field(frame, 'end_position_long')
                    lap_time       = get_frame_field(frame, 'total_timer_time')
                    lap_distance   = get_frame_field(frame, 'total_distance')
                    avg_speed      = get_frame_field(frame, 'avg_speed')
                    
                    if None not in [timestamp, start_pos_lat, start_pos_long, end_pos_lat, 
                                    end_pos_long, lap_time, lap_distance, avg_speed]:
                        entry = pd.DataFrame([{'Timestamp':      timestamp, 
                                               'start_pos_lat':  start_pos_lat,
                                               'start_pos_long': start_pos_long, 
                                               'end_pos_lat':    end_pos_lat,
                                               'end_pos_long':   end_pos_long,
                                               'lap_time':       lap_time,
                                               'lap_distance':   lap_distance,
                                               'avg_speed':      avg_speed}])
                        laps = pd.concat([laps, entry], axis=0, ignore_index=True)
                        
                    
                elif frame.name == 'record':
                    timestamp    = get_frame_field(frame, 'timestamp')
                    lat          = get_frame_field(frame, 'position_lat')
                    long         = get_frame_field(frame, 'position_long')
                    speed        = get_frame_field(frame, 'speed')
                    cadence      = get_frame_field(frame, 'cadence')
                    frac_cadence = get_frame_field(frame, 'fractional_cadence')
                    
                    if None not in [timestamp, lat, long, speed, cadence, frac_cadence]:
                        # cadence calculation:
                        # https://forums.garmin.com/developer/fit-sdk/f/discussion/288454/fractional-cadence-values 
                        entry = pd.DataFrame([{'Timestamp': timestamp, 
                                               'lat':       lat,
                                               'long':      long, 
                                               'speed':     speed,
                                               'cadence':   (cadence + frac_cadence) * 2}])
                        points = pd.concat([points, entry], axis=0, ignore_index=True)
    
    lat_mean = np.cos(points.lat.mean())
    SCALING_FACTOR = 10

    # why we need to divide by 2**32 / 360:
    # https://gis.stackexchange.com/questions/371656/garmin-fit-coordinate-system
    points = (
        points
        .assign(lat =lambda x: x.lat / (2**32 / 360))
        .assign(long=lambda x: x.long / (2**32 / 360))

        # project roughly onto x y plane
        # https://stackoverflow.com/questions/16266809/convert-from-latitude-longitude-to-x-y 
        .assign(x=lambda x: SCALING_FACTOR * R * x.long * np.cos(lat_mean))
        .assign(y=lambda x: SCALING_FACTOR * R * x.lat)

        # center the points - need to decide a reasonable place to center (such as center of track)
        # .assign(x=lambda x: x.x - x.x.min())
        # .assign(y=lambda x: x.y - x.y.min())
    )
    
    laps = (
        laps
        .assign(start_pos_lat =lambda x: x.start_pos_lat  / (2**32 / 360))
        .assign(start_pos_long=lambda x: x.start_pos_long / (2**32 / 360))
        .assign(end_pos_lat   =lambda x: x.end_pos_lat    / (2**32 / 360))
        .assign(end_pos_long  =lambda x: x.end_pos_long   / (2**32 / 360))

        .assign(start_x=lambda x: SCALING_FACTOR * R * x.start_pos_long * np.cos(lat_mean))
        .assign(start_y=lambda x: SCALING_FACTOR * R * x.start_pos_lat)
        .assign(end_x  =lambda x: SCALING_FACTOR * R * x.end_pos_long   * np.cos(lat_mean))
        .assign(end_y  =lambda x: SCALING_FACTOR * R * x.end_pos_lat)
    )    
    
    return (points, laps)


def plot_map(points, laps, pc="blue", lc="red", cmap='brg'):
    """
    Inputs: 
        points and laps to be plotted
        pc: colour of normal points. Default blue, can specify variable like "Timestamp" to colour by time
        lc: colour of lap points. Default red, yellow also good if using brg
        cmap: used if there is a range of colours. gray and brg useful gradients

    prints map of points, with options for colouring

    Output: Returns none, but displays plot
    """

    max_diff = max(points.x.max() - points.x.min(), 
                   points.y.max() - points.y.min())
    
    ax = points.plot.scatter(x="x", 
                             y="y",
                             c=pc,
                             s=5,
                             xlim=(points.x.min() - MARGIN, max(points.x.max(), points.x.min() + max_diff) + MARGIN),
                             ylim=(points.y.min() - MARGIN, max(points.y.max(), points.y.min() + max_diff) + MARGIN),
                             cmap=cm.get_cmap(cmap),
                             figsize=(7, 7)
    )

    laps.plot.scatter(x="end_x",
                      y="end_y",
                      alpha=0.7,
                      c=lc,
                      ax=ax)
    
    ax.legend(["Data points", "Laps"])



def plot_map(points, x, y, figsize, ax):
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
    points.plot.scatter(x=x,
                        y=y,
                        s=3,
                        alpha=0.7,
                        c="red",
                        ax=ax)


def normalise_to_mean(points, reference_points=None):
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

        return axis

    else:
        points, reference_points = normalise_to_mean(points, reference_points)
        
        figure, ax = plt.subplots()
        plot_map(points, x, y, (10, 10), ax)

        if reference_points is not None:
            plot_reference(reference_points, x, y, ax)

        if connected:
            points.plot(x, y, ax=ax)

        return ax


