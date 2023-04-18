import pandas as pd
import numpy as np
from random import randint
from track import Track


SAMPLING_STD_DEV = 4
MACRO_STD_DEV = 8
FULL_TRACK_T = 10.907  # parametric value of t for the full track rotation


def random_out_and_back():
    """
    random_out_and_back: generates a random walk out and back. 
    Walk back is different to walk out. 
    Roughly in a straight line

    Outputs ~
        pd.DataFrame of x, y skeleton points defining the corners of the walk
    """
    forward_num_steps = randint(5, 10)
    backward_num_steps = randint(5, 10)
    random_walk = pd.DataFrame(index=range(forward_num_steps + backward_num_steps + 1), columns=["x", "y"])

    random_walk.iloc[0] = (0, 0)

    for i in range(1, forward_num_steps):
        random_walk.iloc[i] = (randint(-50, 500), randint(-100, 100))
    
    for i in range(forward_num_steps, forward_num_steps + backward_num_steps):
        random_walk.iloc[i] = (randint(-500, 50), randint(-100, 100))

    random_walk = (
        random_walk
        .assign(x=lambda x: x.x.cumsum())
        .assign(y=lambda x: x.y.cumsum())
    )

    random_walk.iloc[-1] = (0, 0)

    return random_walk


def out_and_back():
    """
    out_and_back: generates a random walk out and back. 
    Walk back is roughly the same as walk out. 
    Roughly in a straight line

    Outputs ~
        pd.DataFrame of x, y skeleton points defining the corners of the walk
    """
    forward_num_steps = randint(5, 15)
    random_walk = pd.DataFrame(index=range(forward_num_steps * 2 - 1), columns=["x", "y"])

    random_walk.iloc[0] = (0, 0)

    for i in range(1, forward_num_steps):
        rand_vect = (randint(-50, 500), randint(-100, 100))
        random_walk.iloc[i] = rand_vect
        random_walk.iloc[-i] = (np.random.normal(loc=-rand_vect[0], scale=MACRO_STD_DEV), 
                                np.random.normal(loc=-rand_vect[1], scale=MACRO_STD_DEV))
    
    random_walk = (
        random_walk
        .assign(x=lambda x: x.x.cumsum())
        .assign(y=lambda x: x.y.cumsum())
    )

    return random_walk


def loop():
    """
    loop: generates a random loop.
    3-4 corners, makes either a triangle or a square

    Outputs ~
        pd.DataFrame of x, y skeleton points defining the corners of the walk
    """
    num_steps = randint(3, 5)

    random_walk = pd.DataFrame(index=range(num_steps + 1), columns=["x", "y"])
    random_walk.iloc[0] = (0, 0)

    cum_angle = 0

    for i in range(1, num_steps):
        cum_angle += np.pi / 4 + np.random.random() * np.pi / 2
        length = randint(200, 1000)

        random_walk.iloc[i] = (length * np.cos(cum_angle), length * np.sin(cum_angle))

    random_walk = (
        random_walk
        .assign(x=lambda x: x.x.cumsum())
        .assign(y=lambda x: x.y.cumsum())
    )

    random_walk.iloc[-1] = (0, 0)

    return random_walk


def small_loop():
    """
    small_loop: generates a set of random small loops.

    Outputs ~
        pd.DataFrame of x, y skeleton points defining the corners of the walk
    """
    num_laps = randint(3, 5)

    CORNERS = [(70, 100), (70, -250), (-70, -250), (-70, 100)]

    random_walk = pd.DataFrame(index=range(num_laps * 4), columns=["x", "y"])

    for i in range(0, 4 * num_laps):
        point = CORNERS[i % 4]
        random_walk.iloc[i] = (np.random.normal(loc=point[0], scale=MACRO_STD_DEV), 
                               np.random.normal(loc=point[1], scale=MACRO_STD_DEV))

    return random_walk


def circular_warm_up():
    """
    
    """
    SCALER = 1.4
    track, _, _ = generate_track(8)

    return (
        randomise_location(track)
        .assign(x=lambda df: df.x * SCALER)
        .assign(y=lambda df: df.y * SCALER)
    )



def sample(df):
    """
    sample: given a df of skeleton points, sample a point every STEP_SIZE meters

    Inputs ~
        df: pd.DataFrame with columns x, y denoting the skeleton points

    Outputs ~
        points: pd.DataFrame with points interpolated and sampled every STEP_SIZE meters
    """
    STEP_SIZE = 20

    points = pd.DataFrame(columns=["x", "y"])
    for i in range(1, len(df)):
        new_points = []
        diff_vec = (df.iloc[i].x - df.iloc[i-1].x, df.iloc[i].y - df.iloc[i-1].y)
        seg_length = (diff_vec[0]**2 + diff_vec[1]**2) ** 0.5
        unit_vec = (diff_vec[0] / seg_length, diff_vec[1] / seg_length)

        for j in range(int(seg_length) // STEP_SIZE):
            new_points.append((df.iloc[i-1].x + j * STEP_SIZE * unit_vec[0], df.iloc[i-1].y + j * STEP_SIZE * unit_vec[1]))

        points = pd.concat([points, pd.DataFrame(new_points, columns = ["x", "y"])])

    return points


def randomise_location(df):
    """
    randomise_location: given a df of coordinates, add Gaussian noise to the points

    Inputs ~
        df: pd.DataFrame with columns x, y of points to randomise

    Outputs ~ 
        df: pd.DataFrame of randomised points
    """
    df.x = np.random.normal(loc=df['x'], scale=SAMPLING_STD_DEV)
    df.y = np.random.normal(loc=df['y'], scale=SAMPLING_STD_DEV)
    return df


def randomise_rotation(df):
    """
    randomise_rotation: rotates all points around the origin a random amount

    Inputs ~
        df: pd.DataFrame of points to be rotated

    Outputs ~
        df: pd.DataFrame of rotated points
    """
    # https://academo.org/demos/rotation-about-point/ 
    angle = np.random.random() * 2 * np.pi

    c, s = np.cos(angle), np.sin(angle)
    j = np.array([[c, s], [-s, c]])
    m = np.dot(j, [df.x, df.y])

    df.x = m[0]
    df.y = m[1]
    return df


def randomise_points(df):
    """
    randomise_points: first randomly rotate points, then add random noise

    Inputs ~
        df: pd.DataFrame of points to be rotated and randomised

    Outputs ~
        df: pd.DataFrame of randomised points
    """
    return randomise_location(randomise_rotation(df))


def generate_400_section():
    """
    generate_400_section: randomly sample points from a 400 track
    This is done by stepping forward around the track in random intervals

    Outputs ~
        df: pd.DataFrame of points in a 400m loop. These have not had noise added
    """
    track = Track()

    track_points = []
    t = 0

    while t < FULL_TRACK_T:
        t += 0.4 + 0.2 * np.random.random()   # increase by between [0.4, 0.6] radians
        track_points.append(track.parametric_point(t))

    return pd.DataFrame(track_points, columns=["x", "y"])
    # # generate 400 section
    # track1 = "activities/4909301731.fit"  # track with warm up around a bigger field, 5th Jan 2021

    # points, laps = read_activity("data/" + track1)

    # _400_lap = (
    #     points
    #     [lambda x: x.Timestamp.between(laps.iloc[2].Timestamp, laps.iloc[3].Timestamp)]
    #     .assign(x=lambda x: x.x - x.x.mean())
    #     .assign(y=lambda x: x.y - x.y.mean())
    # )

    # return _400_lap


def generate_200_section():
    # track2 = "activities/5431221291.fit"  # track with 200's 

    # points, laps = read_activity("data/" + track2)

    # _200_lap = (
    #     points
    #     [lambda x: x.Timestamp.between(laps.iloc[3].Timestamp, laps.iloc[5].Timestamp)]
    #     .assign(x=lambda x: x.x - x.x.mean())
    #     .assign(y=lambda x: x.y - x.y.mean())
    # )

    # return _200_lap
    return None


def generate_track(UPPER_LIMIT=16):
    """
    generate_track: generate random points in the shape of a track
    This is done by generating a random number of 400m or 200m sections

    Outputs ~
        random_walk: pd.DataFrame of all the generated x, y coordinates
        reference_points: pd.DataFrame of reference points. This includes the center point, and the northmost point
        bbox: pd.DataFrame of the bounding box for the track 
    """
    num_laps = randint(2, UPPER_LIMIT)

    random_walk = pd.DataFrame(columns=["x", "y"])

    _200_lap = generate_200_section()

    if np.random.random() < 0.0:  # generate 200m workout
        section = _200_lap
    else:  # generate 400m workout
        for i in range(num_laps):
            random_walk = pd.concat([random_walk, randomise_location(generate_400_section())])
    
    track = Track()
    reference_points = [(0, 0), (0, track.s / 2), (0, -track.s / 2)]
    bbox = [(-track.r, -track.s / 2 - track.r), (-track.r, track.s / 2 + track.r),
            ( track.r,  track.s / 2 + track.r), (track.r, -track.s / 2 - track.r)]

    random_values = randomise_rotation(pd.concat([random_walk, 
                                                  pd.DataFrame(reference_points, columns=["x", "y"]),
                                                  pd.DataFrame(bbox, columns=["x", "y"])]))

    random_walk      = random_values.iloc[:-7]
    reference_points = random_values.iloc[-7:-4]
    bbox             = random_values.iloc[-4:]
    
    bounding_box = [(bbox.x.min(), bbox.y.max()), (bbox.x.max(), bbox.y.min())]
    bbox = pd.DataFrame(bounding_box, columns=["x", "y"])

    # choose the north most point, drop the southern point
    if reference_points.iloc[1].y > reference_points.iloc[2].y:
        reference_points = reference_points.drop(2)
    else:
        reference_points = reference_points.drop(1)

    return (random_walk, reference_points, bbox)


def generate_warm_up():
    """
    generate_warm_up: randomly generate one of the defined warm up sections

    Outputs ~
        points: pd.DataFrame of x, y coordinates
    """
    rand_num = np.random.random()

    if rand_num < 0.2:  # random out and back
        points = random_out_and_back()
    elif rand_num < 0.4: 
        points = out_and_back()
    elif rand_num < 0.6:
        points = small_loop()
    elif rand_num < 0.8:
        points = circular_warm_up()
    else:
        points = loop()
    
    return points


def random_map():
    """
    random_map: randomly generate a run's worth of points
    Includes a track section and warm up section(s)

    Outputs ~
        dict of pd.DataFrames. Keys:
            random_walk: pd.DataFrame of all the generated x, y coordinates
            reference_points: pd.DataFrame of reference points. This includes the center point, and the northmost point
            bbox: pd.DataFrame of the bounding box for the track 
    """
    track, reference_points, bbox = generate_track()

    warm_up_skeleton = generate_warm_up()

    rand_num = np.random.random()
    if rand_num < 0.2:  # different warm up/ cool down, add another random section
        warm_up_skeleton = pd.concat([warm_up_skeleton.iloc[:-1], generate_warm_up()])

    warm_up_points = randomise_points(sample(warm_up_skeleton))
    
    return {"points": pd.concat([track, warm_up_points]), "keypoints": reference_points, "bbox": bbox}