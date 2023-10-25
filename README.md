# COSC470 - BSc Computer Science Honours Research Project

Gareth Harcombe's Computer Science Honours Project - The increased use of GPS watches in the sport of athletics has made widespread analysis of running work-outs from GPS data possible. This project presents several methods for the novel problem of predicting lap times for high-performance athletes in running workouts. There are several steps to this process, starting with collating the data to train and test such a system, and ensuring that the data collected is accurate. After collecting the data, it must be preprocessed. This includes using UTM to project coordinates from latitude and longitude into meter space. This report proposes three methods of locating the athletics track, including CNNs and numeric optimisation, and achieves an acceptable level of error at 4.20 meters for downstream tasks using Convolutional Kernels. After locating the athletics track, GPS points are normalised by mathematically projecting them onto the athletics track, providing both error correction and feature engineering.

All data is then used to produce predictions of lap times using acceleration heuristic approaches and rolling window classifications. These methods both produce accurate predictions and are a good starting point for this problem, although have false positives and false negatives respectively that hamper the performance of both methods. Neither method performs significantly better on all metrics, although the rolling window classification approach showed more potential than the heuristic approach, greatly benefiting from more training data.

## Installation
Clone the repository to your local machine. All data, models, and frameworks are included. Install the required dependencies using ```requirements.txt```.

```bash
pip install -r requirements.txt
```

## Usage
```inference.ipynb``` gives a full model inference with an example workout. This starts by reading an activity's data using the ```dataloader.py``` functions, which reads the data from the specified file and preprocesses the data. The GPS point data is then fed into a prediction model from ```models.py``` to produce a list of predicted lap times. Predicted lap times are evaluated against the ground truth lap times using evaluation metrics from ```eval.py```. 

```lap_times.py``` completes training and testing of the lap time prediction models. This follows a similar pipeline to the above ```inference.ipynb```.

A large sub-problem for this project was locating the athletics track within a workout. The majority of files for this problem are found in the ```track_location``` folder, which includes: a mathematical definition of the track in ```track.py```; track data plotting and transformation tools in ```track_tools.py```; synthetic data generation in ```random_map_generation.py```, with examples in ```random_map_generation.ipynb```; convolution and numeric optimisation methods in ```convolution.py``` and ```optimisation.py``` respectively; and the deep learning keypoint detection model, powered by ```engine.py```, ```utils.py```, and ```keypoint_detection_track.ipynb```. All three methods are compared in ```eval_track.py``` in the home directory. 

There are several notebooks and files that were used in data exploration and visualisation. These include: ```inferenceAcceleration.ipynb``` for testing acceleration heuristic-based lap prediction models; ```data_exploration.ipynb``` for a range of initial data exploration, such as checking available data, plotting functionality, inspecting specific laps within a workout, and numeric optimisation heuristics; and ```check_laps.ipynb``` for testing workout splicing to fix incorrect ground truth lap data.

## Contributing
Due to the assessment nature of this project, please do not contribute to this project without prior approval. 

## Authors and acknowledgment
Project by Gareth Harcombe, supervised by Dr. Kourosh Neshatian. Contributions by Dr. Richard Green for athletics track location. 
