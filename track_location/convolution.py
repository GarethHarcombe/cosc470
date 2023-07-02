import cv2
import numpy as np
from track import Track
from scipy.ndimage import rotate


DIMENSIONS = (60, 60)
TRACK = Track(r=36.5 / 5, s=84.39 / 5)

def load_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    image = cv2.bitwise_not(image)
    image = image[150:850, 150:850]
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return image

def rotate_matrix(matrix, theta):
    # Rotate the matrix using scipy.ndimage.rotate
    rotated_matrix = rotate(matrix, theta, reshape=False, order=1)
    rotated_matrix[rotated_matrix==0.0] = -0.1
    return rotated_matrix


def make_kernel(theta):
    kernel = np.zeros(DIMENSIONS)
    
    for t in range(109):
        track_point = TRACK.parametric_point(t / 10)
        kernel[int(np.round(track_point[0])) + int(DIMENSIONS[0]/2), int(np.round(track_point[1])) + int(DIMENSIONS[1]/2)] = 1

    kernel1 = np.ones((3, 3),np.uint8)
    kernel = cv2.dilate(kernel, kernel1, iterations=1)
    
    kernel[kernel==0] = -0.05
    kernel[kernel==1] = 0.1
    
    kernel = rotate_matrix(kernel, theta)
    return kernel


def average_convolution_output(matching):
    contours, _ = cv2.findContours(matching, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate the average pixel coordinates
    total_x = 0
    total_y = 0
    count = 0

    for contour in contours:
        for point in contour:
            x, y = point[0]
            total_x += x
            total_y += y
            count += 1

    if count > 0:
        average_x = total_x / count
        average_y = total_y / count
        return (average_x, average_y)
    else:
        return None

def predict_track_location(filepath):
    image = load_image(filepath)
    # cv2.imshow('Original', image)

    centers = []
    
    for theta in range(0, 180, 5):
        kernel = make_kernel(theta)
        # cv2.imshow('Kernel'+str(theta), kernel)

        matching = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        cv2.imshow('Convolution', matching)

        center = average_convolution_output(matching)
        if center is not None:
            print(f"Average pixel coordinates with angle {theta}: {center[0]}, {center[1]}")
            centers.append(center)
        else:
            print(f"Angle {theta} has no pixels above threshold")

    if len(centers) > 0:
        location = (sum([center[0] for center in centers])/len(centers), 
                    sum([center[1] for center in centers])/len(centers))
    else:
        location = None

    while(1):
        if cv2.waitKey(33) == ord('a'):
            cv2.destroyAllWindows()
            break

    return location
    


if __name__ == "__main__":
    location = predict_track_location('/home/gareth/Documents/Uni/2023/cosc470/track_location/random_image.png')
    print(f"Average location is {location}")