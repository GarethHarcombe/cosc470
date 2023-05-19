import numpy as np
import pandas as pd
import datetime as dt
from datetime import timezone
import fitdecode
import sweat
import utm
from optimisation import track_location, add_track_cols, transform_points


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


def add_track_features(points):
    res = track_location(points)
    x, y, theta = res.x
    new_points = transform_points(points, x, y, theta)

    return add_track_cols(new_points), res.x


def read_activity(activity):
    if activity[-3:] == "fit":
        return read_fit(activity)
    elif activity[-3:] == "tcx":
        return read_tcx(activity)
    elif activity[-4:] == "xlsx":
        return read_xlsx(activity)


def read_tcx(activity):
    points = pd.DataFrame(columns=['Timestamp', 'lat', 'long', 'speed', 'cadence'])
    laps = pd.DataFrame(columns=['Timestamp', 'start_pos_lat', 'start_pos_long', 'end_pos_lat',
                                 'end_pos_long', 'total_timer_time', 'total_distance', 'avg_speed'])
    
    data = sweat.read_tcx(HOME_DIR + activity)

    points = data[data.lap == 0]
    laps   = data[data.lap == 1]

    return (points, laps)


def read_xlsx(activity):
    points = pd.read_excel(HOME_DIR + activity, sheet_name="points")
    laps   = pd.read_excel(HOME_DIR + activity, sheet_name="laps")

    return (points, laps)


def get_frame_field(frame, field):
    """Checks if the frame has a specific field, returns None if not"""
    if frame.has_field(field):
        return frame.get_value(field)
    return None


def to_xy(lats, longs):
    lst = []
    for lat, long in zip(lats, longs):
        lst.append(utm.from_latlon(lat, long)[:2])
    return pd.DataFrame(lst, columns=["x", "y"])


def read_fit(activity):
    """
    Input: a string directory of an activity file
    Reads all the relevant fields for data points and lap times
    Output: a tuple of datapoints and laps df
    """
    points = pd.DataFrame(columns=['Timestamp', 'lat', 'long', 'speed', 'cadence'])
    laps = pd.DataFrame(columns=['Timestamp', 'start_pos_lat', 'start_pos_long', 'end_pos_lat',
                                 'end_pos_long', 'total_timer_time', 'total_distance', 'avg_speed'])
    events = pd.DataFrame(columns=['Timestamp', 'event', 'event_type'])
    
    with fitdecode.FitReader(HOME_DIR + activity) as fit_file:
        # useful visualisation tool of available frames and fields: https://www.fitfileviewer.com/
        for frame in fit_file:         
            if isinstance(frame, fitdecode.records.FitDataMessage):
                
                if frame.name == 'event':
                    # event_type = start   for start time
                    timestamp  = get_frame_field(frame, 'timestamp')
                    event      = get_frame_field(frame, 'event')
                    event_type = get_frame_field(frame, 'event_type')

                    if None not in [timestamp, event, event_type]:
                        entry = pd.DataFrame([{'Timestamp':  timestamp,
                                               'event':      event,
                                               'event_type': event_type}])
                        events = pd.concat([events, entry], axis=0, ignore_index=True)


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

    try:
        start_time = events[events.event_type == "start"].Timestamp.values[0]
    except IndexError:
        start_time = points.Timestamp.values[0].tz_localize(timezone.utc)

    # why we need to divide by 2**32 / 360:
    # https://gis.stackexchange.com/questions/371656/garmin-fit-coordinate-system
    points, track_params = add_track_features(
        points
        .assign(speed=lambda x: x.speed.astype(float))
        .assign(lat =lambda x: x.lat / (2**32 / 360))
        .assign(long=lambda x: x.long / (2**32 / 360))

        .assign(x=lambda x: to_xy(x.lat, x.long).x)
        .assign(y=lambda x: to_xy(x.lat, x.long).y)

        .assign(Timestamp=lambda df: pd.to_datetime(df.Timestamp).dt.tz_convert(timezone.utc))
        .assign(start_time=lambda df: pd.to_datetime(start_time).tz_convert(timezone.utc))
        .assign(time_after_start=lambda df: (df.Timestamp - df.start_time) / np.timedelta64(1, 's'))
        .drop(columns=["lat", "long", "start_time"])
    )

    
    laps = (
        laps
        .assign(start_pos_lat =lambda x: x.start_pos_lat  / (2**32 / 360))
        .assign(start_pos_long=lambda x: x.start_pos_long / (2**32 / 360))
        .assign(end_pos_lat   =lambda x: x.end_pos_lat    / (2**32 / 360))
        .assign(end_pos_long  =lambda x: x.end_pos_long   / (2**32 / 360))

        .assign(start_x=lambda x: to_xy(x.start_pos_lat, x.start_pos_long).x)
        .assign(start_y=lambda x: to_xy(x.start_pos_lat, x.start_pos_long).y)
        .assign(end_x  =lambda x: to_xy(x.end_pos_lat,   x.end_pos_long).x)
        .assign(end_y  =lambda x: to_xy(x.end_pos_lat,   x.end_pos_long).y)

        .assign(time_after_start=lambda df: df.lap_time.cumsum())
    )    

    x, y, theta = track_params
    laps = transform_points(laps, x, y, theta, "start_x", "start_y")
    laps = transform_points(laps, x, y, theta, "end_x", "end_y")
    
    return (points, laps, events)