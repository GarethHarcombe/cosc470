import cv2
import numpy as np
from .track import Track
from scipy.ndimage import rotate

# on this computer, track is 33 pixels high, so need that plus a small buffer
DIMENSIONS = (36, 36)
KERNEL_SIZE = DIMENSIONS[0] * DIMENSIONS[1]
TRACK_SCALE = 4.77  # (84.39+2*36.5) / 33 = 4.77
TRACK = Track(r=36.5 / TRACK_SCALE, s=84.39 / TRACK_SCALE)
MARGIN = 150

def process_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.bitwise_not(image)
    image = image[MARGIN:1000-MARGIN, MARGIN:1000-MARGIN]
    _, image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY)
    return image

def rotate_matrix(matrix, theta):
    # Rotate the matrix using scipy.ndimage.rotate
    rotated_matrix = rotate(matrix, theta, reshape=False, order=1)
    return rotated_matrix


def make_kernel(theta):
    kernel = np.zeros(DIMENSIONS)
    
    for t in range(109):
        track_point = TRACK.parametric_point(t / 10)
        kernel[int(np.round(track_point[0])) + int(DIMENSIONS[0]/2), int(np.round(track_point[1])) + int(DIMENSIONS[1]/2)] = 1

    kernel1 = np.ones((3, 3),np.uint8)
    kernel = cv2.dilate(kernel, kernel1, iterations=1)
    
    kernel = rotate_matrix(kernel, theta)
    # cv2.imshow('kernel'+str(theta), kernel)
    kernel[kernel==0] = -1 / KERNEL_SIZE
    kernel[kernel >0] =  1 / (KERNEL_SIZE / 5)
    return kernel


def predict_track_location(image):
    image = process_image(image)
    # cv2.imshow('Original', image)
    
    best_match = (0, None, None)   # (highest value, matching, theta)

    for theta in range(0, 180, 3):
        kernel = make_kernel(theta)

        matching = cv2.filter2D(src=image, ddepth=-1, kernel=kernel)
        # cv2.imshow('Convolution'+str(theta), matching)

        if np.max(matching) > best_match[0]:
            best_match = (np.max(matching), matching, theta)

    # print(best_match)
    matrix_indices = np.unravel_index(np.argmax(best_match[1], axis=None), best_match[1].shape)
    matrix_indices = (matrix_indices[1], matrix_indices[0])
    # print(matrix_indices)
    # best_match = (best_match[0], cv2.circle(best_match[1], matrix_indices, 1, (255, 0, 0), 5), best_match[2])
    # cv2.imshow('Best Match'+str(best_match[2]), best_match[1])
    # while(1):
    #     if cv2.waitKey(33) == ord('a'):
    #         cv2.destroyAllWindows()
    #         break
    
    return (matrix_indices[0] + MARGIN, matrix_indices[1] + MARGIN, (best_match[2] - 90) * np.pi / 180)
    


if __name__ == "__main__":
    image = cv2.imread('/home/gareth/Documents/Uni/2023/cosc470/track_location/random_image.png')
    cv2.imshow("Real Original", image)
    print(image.shape)
    location = predict_track_location(image)
    print(f"Average location is {location}")