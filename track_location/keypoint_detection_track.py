# https://medium.com/@alexppppp/how-to-train-a-custom-keypoint-detection-model-with-pytorch-d9af90e111da

import cv2, numpy as np, matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.models.detection.rpn import AnchorGenerator

# https://github.com/pytorch/vision/tree/main/references/detection
from utils import collate_fn
from engine import train_one_epoch, evaluate

from random_map_generation import random_map
from tools import consistent_scale_plot


MAP_SIZE = 2000


def get_margins(image, IMG_WIDTH, IMG_HEIGHT):
    """Find the margins of the image. 
    Used to translate pixel coordinates to meter coordinates """
    image = (image.permute(1,2,0).numpy() * 255).astype(np.uint8)
    row = image[400]

    first_black = np.where(row < 255)[0][0]
    left_margin = (np.where(row[first_black:] == 255) + first_black)[0][0]

    last_black = np.where(row < 255)[0][-1]
    right_margin = (IMG_WIDTH - np.where(row[:last_black] == 255)[0])[-1]

    column = np.array([image[i][450] for i in range(image.shape[0])])
    first_black = np.where(column < 255)[0][0]
    top_margin = (np.where(column[first_black:] == 255) + first_black)[0][0]

    last_black = np.where(column < 255)[0][-1]
    bottom_margin = (IMG_HEIGHT - np.where(column[:last_black] == 255)[0])[-1]

    return {"LEFT": left_margin, "RIGHT": right_margin, "TOP": top_margin, "BOTTOM": bottom_margin}


class ClassDataset(Dataset):
    def __init__(self, length=64):                
        self.length = length
        points = random_map()["points"]
        
        img = self.image_tensor_from_points(points)
        self.MARGINS = get_margins(img, self.IMG_HEIGHT, self.IMG_WIDTH)


    def image_tensor_from_points(self, points):
        """Generate a tensor from a set of GPS points.
        Generates image, then converts image into a tensor for input into the model"""
        fig = consistent_scale_plot(points, MAP_SIZE=MAP_SIZE)
        fig.canvas.draw()
        np_fig = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)

        self.IMG_HEIGHT = int(np.sqrt(len(np_fig) / 3))
        self.IMG_WIDTH = self.IMG_HEIGHT

        img = torch.Tensor([np_fig[0::3], np_fig[1::3], np_fig[2::3]])   # needs to be in C, H, W format. Image is 1000x1000. Colour RGB
        img = torch.reshape(img, (3, self.IMG_HEIGHT, self.IMG_WIDTH)) / 255

        plt.clf()
        plt.close()

        return img


    def x_point_to_image_coords(self, x, points_mean):
        """Convert x coordinate from GPS points into pixel coordinates in the image"""
        CANVAS_WIDTH  = self.IMG_WIDTH  - self.MARGINS["LEFT"] - self.MARGINS["RIGHT"]
        return self.MARGINS["LEFT"] + CANVAS_WIDTH // 2 + (x - points_mean) * (CANVAS_WIDTH   // 2) / MAP_SIZE


    def y_point_to_image_coords(self, y, points_mean):
        """Convert y coordinate from GPS points into pixel coordinates in the image"""
        CANVAS_HEIGHT = self.IMG_HEIGHT - self.MARGINS["TOP"]  - self.MARGINS["BOTTOM"]
        return self.MARGINS["TOP"] + CANVAS_HEIGHT // 2 + (-y + points_mean) * (CANVAS_HEIGHT // 2) / MAP_SIZE


    def __getitem__(self, idx):
        """Generate a random image with keypoints and bounding box"""
        map = random_map()

        # Convert keypoints into a tensor
        keypoints = map["keypoints"]
        keypoints = torch.as_tensor([[[self.x_point_to_image_coords(x, map["points"].x.mean()), 
                                       self.y_point_to_image_coords(y, map["points"].y.mean()), 1] 
                                      for x, y in zip(keypoints.x.values, keypoints.y.values)]], 
                                    dtype=torch.float32)

        # Generate tensor of the GPS points
        img = self.image_tensor_from_points(map["points"])
        
        # Convert bounding box points into a tensor
        bboxes = map["bbox"]
        bboxes = torch.as_tensor([[self.x_point_to_image_coords(bboxes.x.values[0], map["points"].x.mean()), 
                                   self.y_point_to_image_coords(bboxes.y.values[0], map["points"].y.mean()), 
                                   self.x_point_to_image_coords(bboxes.x.values[1], map["points"].x.mean()), 
                                   self.y_point_to_image_coords(bboxes.y.values[1], map["points"].y.mean())]], 
                                dtype=torch.float32)        
        
        # Convert everything into a torch tensor        
        bboxes = torch.as_tensor(bboxes, dtype=torch.float32)       
        target = {}
        target["boxes"] = bboxes
        target["labels"] = torch.as_tensor([1 for _ in bboxes], dtype=torch.int64) # all objects are track
        target["image_id"] = torch.tensor([idx])
        target["area"] = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        target["iscrowd"] = torch.zeros(len(bboxes), dtype=torch.int64)
        target["keypoints"] = keypoints

        return img, target
    
    def __len__(self):
        # This affects how many images are used in training and testing
        return self.length
    

keypoints_classes_ids2names = {0: 'Center', 1: 'North Point'}


def visualize(image, bboxes, keypoints):
    """Visualise map images with keypoints and bounding boxes"""
    fontsize = 15
    
    for bbox in bboxes:
        start_point = (bbox[0], bbox[1])
        end_point = (bbox[2], bbox[3])
        image = cv2.rectangle(image.copy(), start_point, end_point, (0,255,0), 2)
    
    for kps in keypoints:
        for idx, kp in enumerate(kps):
            if keypoints_classes_ids2names[idx] == "Center":
                image = cv2.circle(image.copy(), tuple(kp), 4, (255,0,0), -1)
                kp[1] += 20
                image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
            else:
                image = cv2.circle(image.copy(), tuple(kp), 4, (255,0,0), -1)
                image = cv2.putText(image.copy(), " " + keypoints_classes_ids2names[idx], tuple(kp), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
    
    plt.figure(figsize=(15, 15))
    plt.imshow(image)



def get_model(num_keypoints, weights_path=None):
    # Download model 
    anchor_generator = AnchorGenerator(sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.25, 0.5, 0.75, 1.0, 2.0, 3.0, 4.0))
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=False,
                                                                   pretrained_backbone=True,
                                                                   num_keypoints=num_keypoints,
                                                                   num_classes = 2, # Background is the first class, object is the second class
                                                                   rpn_anchor_generator=anchor_generator)

    if weights_path:
        state_dict = torch.load(weights_path)
        model.load_state_dict(state_dict)        
        
    return model