from torch.utils.data import Dataset, DataLoader
import torch
import os
import fitdecode
import gzip
import pandas as pd
from pathlib import Path
from typing import Any, Callable, Optional
import numpy as np


torch.manual_seed(0)
RUN_TYPES = {"long_run": [1, 0, 0], "workout": [0, 1, 0], "other": [0, 0, 1]}
NUM_POINTS = 60 * 60 * 6
MAX_DIST = 1e6
MAX_SPEED = 15


class Point:
    def __init__(self, timestamp, lat, long, speed):
        self.timestamp = timestamp
        self.lat = lat
        self.long = long
        self.speed = speed
        
    def __repr__(self):
        return f'Point at lat: {self.lat}, long: {self.long}, at time {self.timestamp}, m/s: {self.speed} \n'


class GPSDataset(Dataset):
    """GPS Dataset"""
    
    def __init__(self,
                 root: str,
                 csv_dir: str,
                 transforms: Optional[Callable] = None,
                 fraction: float = 0.2,
                 subset: str = None) -> None:
        """
        Args:
            root (str): Root directory path.
            csv_dir (str): CSV with GPS data from Strava
            transforms (Optional[Callable], optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.ToTensor`` for images. Defaults to None.
            fraction (float, optional): A float value from 0 to 1 which specifies the validation split fraction. Defaults to 0.2.
            subset (str, optional): 'Train' or 'Test' to select the appropriate set. Defaults to None.
        Raises:
            OSError: If csv file doesn't exist in root.
            ValueError: If subset is not either 'Train' or 'Test'
        """
        super().__init__()
        self.root = root
        csv_path = Path(self.root) / csv_dir
        if not csv_path.exists():
            raise OSError(f"{csv_path} does not exist.")

        if subset not in ["Train", "Test"]:
            raise (ValueError(
                f"{subset} is not a valid input. Acceptable values are Train and Test."
            ))
        
        self.fraction = fraction
        self.gps_frame = pd.read_csv(csv_path)
        self.gps_frame = self.gps_frame[self.gps_frame["run_type"].notna()]
        self.gps_frame = self.gps_frame[self.gps_frame.Filename != '']
        self.gps_frame = self.gps_frame[self.gps_frame["Filename"].notna()]
        self.gps_frame = self.gps_frame.filter(["Filename", "run_type"])

        if subset == "Train":
            self.gps_frame = self.gps_frame[:int(
                np.ceil(len(self.gps_frame) * (1 - self.fraction)))]
        else:
            self.gps_frame = self.gps_frame[
                int(np.ceil(len(self.gps_frame) * (1 - self.fraction))):]
            
        for _, row in self.gps_frame.iterrows():
            with gzip.open(os.path.join(self.root, row["Filename"]), 'rb') as f:
                file_content = f.read()
                with open(os.path.join(self.root, row["Filename"][:-3]), 'wb') as w:
                    w.write(file_content)        

    def __len__(self) -> int:
        return len(self.gps_frame)

    def __getitem__(self, idx: int) -> Any:
        
        gps_path = os.path.join(self.root, self.gps_frame.iloc[idx]["Filename"])[:-3]
        points = []
        with fitdecode.FitReader(gps_path) as fit_file:
            for frame in fit_file:
                if isinstance(frame, fitdecode.records.FitDataMessage):
                    if frame.name == 'lap':
                        #print(frame.get_value('total_elapsed_time'), frame.get_value('avg_speed'), frame.get_value('total_distance'), frame.get_value('start_time'))
                        pass
                        
                    elif frame.name == 'record':
                        if frame.has_field('position_lat') and frame.has_field('position_long') and frame.has_field('timestamp') and frame.has_field('speed'):
                            if frame.get_value('timestamp') is not None and frame.get_value('position_lat') is not None and frame.get_value('position_long') is not None and frame.get_value('speed') is not None:
                                point = Point(frame.get_value('timestamp'), frame.get_value('position_lat'), frame.get_value('position_long'), frame.get_value('speed'))
                                points.append(point)        

        lats = [(points[0].lat - point.lat) / MAX_DIST for point in points] 
        lats = lats + (NUM_POINTS - len(lats)) * [0]

        longs = [(points[0].long - point.long) / MAX_DIST for point in points]
        longs = longs + (NUM_POINTS - len(longs)) * [0]
        
        timestamps = [(point.timestamp - points[0].timestamp).seconds / NUM_POINTS for point in points]
        timestamps = timestamps + (NUM_POINTS - len(timestamps)) * [0]
        
        speeds = [point.speed / MAX_SPEED for point in points]
        speeds = speeds + (NUM_POINTS - len(speeds)) * [0]
        
        run_type = self.gps_frame.iloc[idx]["run_type"]

        sample = {'data': torch.tensor([lats, longs, speeds, timestamps]), 'type': torch.FloatTensor(RUN_TYPES[run_type])}
        return sample

        
def get_dataloaders(root_dir, csv_file, batch_size=8):
    gps_datasets = {x: GPSDataset(root_dir, csv_file, subset=x) for x in ["Train", "Test"]}  

    #print(gps_datasets["Train"][622])
    dataloaders = {
        x: DataLoader(gps_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=4)
        for x in ['Train', 'Test']
    }
    return dataloaders
        
        
if __name__ == "__main__":
    dataloaders = get_dataloaders("/home/gareth/Documents/Programming/running/export_12264640/", "activities.csv")
    
    it = iter(dataloaders["Train"])
    print(next(it))
