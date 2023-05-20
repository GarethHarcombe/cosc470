# COSC428 - Identifying the location and orientation of Athletics Track

## How to use
keypoint_detection_track.ipynb contains a full runthrough of the script, including visualisation tools, and loading, training and testing the model.


## Code structure
All of the code for generating random tracks is in random_map_generation.py, with random_map_testing.ipynb giving examples for use.

track.py mathematically defines an athletics track, which is used in map generation. 

engine.py contains the training and evaluation loop.

tools.py contains mapping and image tools.

utils.py provides general utilities such as Metric loggers.

running_points.csv contains a real example of GPS points from a running workout to compare against

keypointsrcnn_weights.pth contains the weights for a trained CNN model