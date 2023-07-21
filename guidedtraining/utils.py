import torchvision
import torch
from math import *
from captum.attr import visualization as viz

def resize(image, bbox, dims):
    tf = torchvision.transforms.Resize(dims)
    new_image = tf(image)
    new_boxes = bbox
    for i in bbox:
        i[0] *= (dims[0] / image.width)
        i[1] *= (dims[1] / image.height)
        i[2] *= (dims[0] / image.width)
        i[3] *= (dims[1] / image.height)

    return new_image, new_boxes

def predict(input):
    output = input
    print(output.size())
    for i in range (len(output)):
        for j in range(len(output[i])):
            if output[i][j] > 0.65:
                output[i][j] = 1
            else:
                output[i][j] = 0
    
    return output


def transform(image, bbox):
    new_image = image
    new_boxes = bbox

    new_image, new_boxes = resize(new_image, new_boxes, dims=(224, 224))
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    new_image = tf(new_image)
    return new_image, new_boxes

def batch_loc_loss(batch_attr_map, batch_bboxes):
  energy_bbox = 0
  energy_whole = 0

  for i in range(len(batch_attr_map)):
    curr = batch_attr_map[i]
    sum_attr = curr[0] + curr[1] + curr[2]
    sum_attr = torch.from_numpy(viz._normalize_attr(sum_attr, 'absolute_value', 2))
    empty = torch.zeros((224, 224))
    for x1, y1, x2, y2 in batch_bboxes[i]:
        empty[floor(x1):ceil(x2), floor(y1):ceil(y2)] = 1
    mask_bbox = sum_attr * empty  
    
    energy_bbox += mask_bbox.sum()
    energy_whole += sum_attr.sum()
  
  proportion = energy_bbox / energy_whole
  
  return proportion