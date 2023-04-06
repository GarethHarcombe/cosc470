import pandas as pd
import numpy as np
from random import randint
from track import Track


def random_out_and_back():
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
    MEAN = 0
    STD_DEV = 4
    forward_num_steps = randint(5, 15)
    random_walk = pd.DataFrame(index=range(forward_num_steps * 2 - 1), columns=["x", "y"])

    random_walk.iloc[0] = (0, 0)

    for i in range(1, forward_num_steps):
        rand_vect = (randint(-50, 500), randint(-100, 100))
        random_walk.iloc[i] = rand_vect
        random_walk.iloc[-i] = (-rand_vect[0] + np.random.normal(loc=MEAN, scale=STD_DEV), 
                                -rand_vect[1] + np.random.normal(loc=MEAN, scale=STD_DEV))
    
    random_walk = (
        random_walk
        .assign(x=lambda x: x.x.cumsum())
        .assign(y=lambda x: x.y.cumsum())
    )

    return random_walk


def loop():
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


def sample(df):
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
    STD_DEV = 1
    df.x = np.random.normal(loc=df['x'], scale=STD_DEV)
    df.y = np.random.normal(loc=df['y'], scale=STD_DEV)
    return df


def randomise_rotation(df):
    # https://academo.org/demos/rotation-about-point/ 
    angle = np.random.random() * 2 * np.pi

    c, s = np.cos(angle), np.sin(angle)
    j = np.array([[c, s], [-s, c]])
    m = np.dot(j, [df.x, df.y])

    df.x = m[0]
    df.y = m[1]
    return df


def randomise_points(df):
    return randomise_location(randomise_rotation(df))


FULL_TRACK_T = 10.907

def generate_400_section():
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


def generate_track():
    num_laps = randint(2, 16)

    random_walk = pd.DataFrame(columns=["x", "y"])

    _200_lap = generate_200_section()

    if np.random.random() < 0.0:  # generate 200m workout
        section = _200_lap
    else:  # generate 400m workout
        for i in range(num_laps):
            random_walk = pd.concat([random_walk, randomise_location(generate_400_section())])
    
    track = Track()
    reference_points = [(0, 0), (0, track.s / 2), (0, -track.s / 2)]
    bbox = [(track.r, track.s / 2), (-track.r, -track.s / 2 - track.r)]

    random_values = randomise_rotation(pd.concat([random_walk, 
                                                  pd.DataFrame(reference_points, columns=["x", "y"]),
                                                  pd.DataFrame(bbox, columns=["x", "y"])]))

    random_walk      = random_values.iloc[:-5]
    reference_points = random_values.iloc[-5:-2]
    bbox             = random_values.iloc[-2:]

    # choose the north most point, drop the southern point
    if reference_points.iloc[1].y > reference_points.iloc[2].y:
        reference_points = reference_points.drop(2)
    else:
        reference_points = reference_points.drop(1)

    return (random_walk, reference_points, bbox)


def generate_warm_up():
    rand_num = np.random.random()

    if rand_num < 0.3:  # random out and back
        points = random_out_and_back()
    elif rand_num < 0.6: 
        points = out_and_back()
    else:
        points = loop()
    
    return points


def random_map():
    track, reference_points, bbox = generate_track()

    warm_up_skeleton = generate_warm_up()

    rand_num = np.random.random()
    if rand_num < 0.2:  # different warm up/ cool down, add another random section
        warm_up_skeleton = pd.concat([warm_up_skeleton.iloc[:-1], generate_warm_up()])

    warm_up_points = randomise_points(sample(warm_up_skeleton))
    
    return {"points": pd.concat([track, warm_up_points]), "keypoints": reference_points, "bbox": bbox}