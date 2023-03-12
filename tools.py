# Gareth 06/03/2023 - Data Visualisation and other tools

import numpy as np
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt
import fitdecode
from scipy.optimize import fsolve

MARGIN = 0.0005


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
    r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
    return c * r


def get_frame_field(frame, field):
    """Checks if the frame has a specific field, returns None if not"""
    if frame.has_field(field):
        return frame.get_value(field)
    return None


def read_activity(activity):
    """
    Input: a string directory of an activity file
    Reads all the relevant fields for data points and lap times
    Output: a tuple of datapoints and laps df
    """
    points = pd.DataFrame(columns=['Timestamp', 'lat', 'long', 'speed', 'cadence'])
    laps = pd.DataFrame(columns=['Timestamp', 'start_pos_lat', 'start_pos_long', 'end_pos_lat',
                                 'end_pos_long', 'total_timer_time', 'total_distance', 'avg_speed'])
    
    with fitdecode.FitReader("/home/gareth/Documents/Uni/2023/cosc470/data/" + activity) as fit_file:
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
    
    # why we need to divide by 2**32 / 360:
    # https://gis.stackexchange.com/questions/371656/garmin-fit-coordinate-system
    points = (
        points
        .assign(lat =lambda x: x.lat / (2**32 / 360))
        .assign(long=lambda x: x.long / (2**32 / 360))
    )
    
    laps = (
        laps
        .assign(start_pos_lat =lambda x: x.start_pos_lat  / (2**32 / 360))
        .assign(start_pos_long=lambda x: x.start_pos_long / (2**32 / 360))
        .assign(end_pos_lat   =lambda x: x.end_pos_lat    / (2**32 / 360))
        .assign(end_pos_long  =lambda x: x.end_pos_long   / (2**32 / 360))
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
    max_diff = max(points.lat.max()  - points.lat.min(), 
                   points.long.max() - points.long.min())
    
    ax = points.plot.scatter(x="lat", 
                        y="long",
                        c=pc,
                        s=5,
                        xlim=(points.lat.min() - MARGIN, max(points.lat.max(), points.lat.min() + max_diff) + MARGIN),
                        ylim=(points.long.min() - MARGIN, max(points.long.max(), points.long.min() + max_diff) + MARGIN),
                        cmap=cm.get_cmap(cmap)
    )

    laps.plot.scatter(x="end_pos_lat",
                      y="end_pos_long",
                      alpha=0.7,
                      c=lc,
                      ax=ax)
    
    ax.legend(["Data points", "Laps"])


class Circle:
    def __init__(self, a, b, r, top_half) -> None:
        """
        circle in form:
          x = a + r cos(t)
          y = b + r sin(t)

        top_half: bool whether the circle is the top half or bottom half of the circle/track
        """
        self.a = a
        self.b = b
        self.r = r

        self.fx = lambda t: self.a + self.r * np.cos(t)
        self.fy = lambda t: self.b + self.r * np.sin(t)
        
        self.dydx = lambda t: -np.cos(t) / (np.sin(t))
        self.top_half = top_half

    def closest_point_on_circle(self, x1, y1):
        """
        Given a point, find the closest projection onto the circle
        Return None if below/above the semi circle defined by self.top_half
        """
        t = np.arctan2(y1 - self.b, x1 - self.a)

        if self.top_half and (t > np.pi or t < 0):
            return None
        elif not self.top_half and t < np.pi and t > 0:
            return None
        else:
            return (self.fx(t), self.fy(t))


def closest_point_on_track(x1, y1):
    """
    Projects any point onto the closest point on a 400m track. 
    Assumes 400m track is centered at (0, 0)
    
    400m track - standard:
    https://www.desmos.com/calculator/enbx7xeuzp 
    there are also tracks with shorter straights (around 80m) and 120m bends - these are ignored for now
    """

    r = 36.5   # radius of the circles
    s = 84.39  # straight lengths

    # first bend
    circle = Circle(0, s/2, r, True)
    first_bend_point = circle.closest_point_on_circle(x1, y1)

    # back straight
    back_straight = (-r, y1) if y1 < s/2 and y1 > -s/2 else None

    # back bend
    circle = Circle(0, -s/2, r, False)
    back_bend_point = circle.closest_point_on_circle(x1, y1)
    
    # home straight
    home_straight = (r, y1) if y1 < s/2 and y1 > -s/2 else None

    points = [first_bend_point, back_straight, back_bend_point, home_straight]
    return [point for point in points if point is not None][0]

