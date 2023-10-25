import numpy as np
import pandas as pd
import datetime as dt
from datetime import timezone
import fitdecode
import sweat
import utm
from torch.utils.data import Dataset, DataLoader
import torch
from typing import Any, Callable, Optional
from track_location.track_tools import add_track_cols, transform_points
from track_location.optimisation import track_location
from track_location.keypoint_detection_track import ClassDataset
import track_location.convolution as convolution


R = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
HOME_DIR = "/home/gareth/Documents/Uni/2023/cosc470/"


def haversine(lat1, long1, lat2, long2):
    """
    haversine: compute the distance between two given GPS coordinates
    Converted to be vectorised from:
    https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points 
    
    Calculate the great circle distance in kilometers between two points 
    on the earth (specified in decimal degrees)

    Inputs ~
        lat1:  float - latitude  of first  point
        long1: float - longitude of first  point
        lat2:  float - latitude  of second point
        long2: float - longitude of second point

    Outputs ~
        distance: float - distance between the two points
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

ConvolutionClass = ConvolutionMethod()

def add_track_features(points, center_ground_truth=False, track=None):
    """
    add_track_features: finds the location of a track and adds relevant features to the input df

    Inputs ~
        points: pd.DataFrame - df of GPS points from a running workout
        center_ground_truth: bool - if true, center around the ground truth track location rather than predicted

    Outputs ~
        (pd.DataFrame, np.array) - df with added features as columns, and array of length 3 with track location and orientation
    """
    # res = track_location(points)
    # x, y, theta = res.x
    # new_points = transform_points(points, x, y, theta)

    # return add_track_cols(new_points), res
    if center_ground_truth:
        res = track
    else:
        res = ConvolutionClass.cv_method(points)
    x, y, theta = res
    
    new_points = transform_points(points, x, y, theta)

    return add_track_cols(new_points), res


def read_activity(activity, center_points=True, track=None):
    """
    read_activity: reads the input file and returns dfs with the data from the file
    Can read .fit, .tcx and .xlsx files

    Inputs ~ 
        activity: str - file name of the file to read

    Outputs ~
        (pd.DataFrame, pd.DataFrame, (optional) pd.DataFrame) - df of GPS points, df of laps, df of events
    """
    if activity[-3:] == "fit":
        return read_fit(activity, center_points=center_points, track=track)
    elif activity[-3:] == "tcx":
        return read_tcx(activity)
    elif activity[-4:] == "xlsx":
        return read_xlsx(activity)


def read_tcx(activity):
    """
    read_tcx: read a given tcx file and return GPS points and laps

    Inputs ~
        activity: str - file path of .tcx file

    Outputs ~
        (pd.DataFrame, pd.DataFrame) - df of GPS points, and df of laps
    """
    points = pd.DataFrame(columns=['Timestamp', 'lat', 'long', 'speed', 'cadence'])
    laps = pd.DataFrame(columns=['Timestamp', 'start_pos_lat', 'start_pos_long', 'end_pos_lat',
                                 'end_pos_long', 'total_timer_time', 'total_distance', 'avg_speed'])
    
    data = sweat.read_tcx(HOME_DIR + activity)

    points = data[data.lap == 0]
    laps   = data[data.lap == 1]

    return (points, laps)


def read_xlsx(activity):
    """
    read_xlsx: read a given xlsx file and return GPS points and laps

    Inputs ~
        activity: str - file path of .xlsx file

    Outputs ~
        (pd.DataFrame, pd.DataFrame) - df of GPS points, and df of laps
    """
    points = pd.read_excel(HOME_DIR + activity, sheet_name="points")
    laps   = pd.read_excel(HOME_DIR + activity, sheet_name="laps")

    return (points, laps)


def get_frame_field(frame, field):
    """
    get_frame_field: given a frame from a fit file, return the field value if it exists. Else return None

    Inputs ~
        frame: fit frame - frame read from a fit file
        field: str - field name to retrieve value from 

    Outputs ~
        (optional) any - the field value from the frame if it exists
    """
    if frame.has_field(field):
        return frame.get_value(field)
    return None


def to_xy(lats, longs):
    """
    to_xy: given a list of latitude and longitudes, convert to x and y coordinates using the UTM scheme

    Inputs ~
        lats:  list of floats - list of latitudes
        longs: list of floats - list of longitudes
    """
    lst = []
    for lat, long in zip(lats, longs):
        lst.append(utm.from_latlon(lat, long)[:2])
    return pd.DataFrame(lst, columns=["x", "y"])


def read_fit(activity, center_points=True, track=None):
    """
    read_fit: read a given fit file and return GPS points and laps

    Inputs ~
        activity: str - file path of .fit file
        center_points: bool - whether to center the points around the center of the track

    Outputs ~
        (pd.DataFrame, pd.DataFrame, pd.DataFrame) - df of GPS points, df of laps, and df of events
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
                    # every routine GPS point recorded
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
        start_time = pd.Timestamp(events[events.event_type == "start"].Timestamp.values[0]).tz_localize(timezone.utc)
    except IndexError:
        start_time = points.Timestamp.values[0].tz_localize(timezone.utc)

    # why we need to divide by 2**32 / 360:
    # https://gis.stackexchange.com/questions/371656/garmin-fit-coordinate-system
    points = (
        points
        # convert coordinates to float value
        .assign(speed=lambda x: x.speed.astype(float))
        .assign(lat =lambda x: x.lat / (2**32 / 360))
        .assign(long=lambda x: x.long / (2**32 / 360))

        # project onto x y plane with meter units
        .assign(x=lambda x: to_xy(x.lat, x.long).x)
        .assign(y=lambda x: to_xy(x.lat, x.long).y)

        # find the time since the start of the GPS recording for each point
        .assign(Timestamp=lambda df: pd.to_datetime(df.Timestamp).dt.tz_convert(timezone.utc))
        .assign(start_time=lambda df: pd.to_datetime(start_time).tz_convert(timezone.utc))
        .assign(time_after_start=lambda df: (df.Timestamp - df.start_time) / np.timedelta64(1, 's'))
        .assign(acceleration=lambda df: (df.speed.diff() / df.Timestamp.diff().dt.total_seconds()).abs())
        .dropna()
        .reset_index(drop=True)
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

    if center_points:
        points, track_params = add_track_features(points, track=track)
        x, y, theta = track_params
        laps = transform_points(laps, x, y, theta, "start_x", "start_y")
        laps = transform_points(laps, x, y, theta, "end_x", "end_y")
    
    return (points, laps, events)



class GPSDataset(Dataset):
    """GPS Dataset"""
    
    def __init__(self,
                 root: str,
                 csv_dir: str,
                 fraction: float = 0.2,
                 subset: str = None) -> None:
        """
        Args:
            root (str): Root directory path.
            csv_dir (str): CSV with GPS data from Strava
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to 0.2.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
        Raises:
            OSError: If csv file doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
        """
        super().__init__()

        self.root = root
        self.csv_dir = csv_dir
        self.fraction = fraction

        csv_path = self.root + self.csv_dir

        # if not csv_path.exists():
        #     raise OSError(f"{csv_path} does not exist.")

        if subset not in ["Train", "Test"]:
            raise (ValueError(
                f"{subset} is not a valid input. Acceptable values are Train and Test."
            ))
        
        activities = pd.read_csv(csv_path)
        self.activities = (
            activities
            [(activities.HAS_ACCURATE_LAPS == 1.0)]
            .reset_index(drop=True)
            [["Activity ID", "Activity Date", "Activity Name", "IS_TRACK_WORKOUT", 
              "HAS_ACCURATE_LAPS", "Activity Type", "Activity Description", "Elapsed Time", 
              "Distance", "Max Heart Rate", "Filename", "Elapsed Time.1", "Moving Time",
              "Distance.1", "Max Speed", "Average Speed", "Elevation Gain"]]
        )

        if subset == "Train":
            self.activities = self.activities[:int(
                np.ceil(len(self.activities) * (1 - self.fraction)))]
        else:
            self.activities = self.activities[
                int(np.ceil(len(self.activities) * (1 - self.fraction))):]
              
    def __len__(self) -> int:
        return len(self.activities)

    def __getitem__(self, idx: int) -> Any:
        filename = self.root + self.activities.iloc[idx].Filename[:-3]
        points, laps, events = read_activity(filename)

        output = {
            "points_speed": points.speed.values,
            "points_cadence": points.cadence.values,
            "points_time": points.time_after_start.values,
            "points_x": points.x.values,
            "points_y": points.y.values,
            "points_dist_to_track": points.dist_to_track.values,
            "lap_times": laps.time_after_start.values
        }
        for key, item in output.items():
            if item.dtype == object:
                # Convert the arrays to a compatible data type
                output[key] = item.astype(float)
            output[key] = torch.tensor(output[key])
        return output
        

        
def get_dataloaders(root_dir, csv_file, batch_size=1):
    gps_datasets = {x: GPSDataset(root_dir, csv_file, subset=x) for x in ["Train", "Test"]}  

    dataloaders = {
        x: DataLoader(gps_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)
        for x in ['Train', 'Test']
    }
    return dataloaders


if __name__ == "__main__":
    dataloaders = get_dataloaders("data/", "activities.csv")
    print("Got dataloaders, now retrieving item")

    it = iter(dataloaders["Train"])
    print(next(it))
