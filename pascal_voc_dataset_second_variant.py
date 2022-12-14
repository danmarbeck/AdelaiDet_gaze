import os
import json
import cv2
import numpy as np
from PIL import Image
import pathlib

path_to_train_image_txt = os.path.join('/data/PascalVOC2012/VOC2012/', 'ImageSets', 'Segmentation', 'val.txt')
path_to_xml_annotation_files = os.path.join('/data/PascalVOC2012/VOC2012/', 'Annotations')

path_to_train_images = os.path.join('/data/PascalVOC2012/VOC2012/', 'JPEGImages')
path_to_segmentation_masks_object_specific = os.path.join('/data/PascalVOC2012/VOC2012/', 'SegmentationObject')
path_to_segmentation_masks_class_specific = os.path.join('/data/PascalVOC2012/VOC2012/', 'SegmentationClass')

path_to_save_save_converted_annotations = os.path.join('/data/PascalVOC2012/VOC2012/random_gaze/val/')


categorie_ids = {0 : 'background',
              1 : 'aeroplane',
              2 : 'bicycle',
              3 : 'bird',
              4 : 'boat',
              5 : 'bottle',
              6 : 'bus',
              7 : 'car',
              8 : 'cat',
              9 : 'chair',
              10 : 'cow',
              11 : 'diningtable',
              12 : 'dog',
              13 : 'horse',
              14 : 'motorbike',
              15 : 'person',
              16 : 'pottedplant',
              17 : 'sheep',
              18 : 'sofa',
              19 : 'train',
              20 : 'tvmonitor'}


images = []
with open(path_to_train_image_txt) as f:
    lines = f.readlines()
    for line in lines:
        images.append(line.split('\n')[0])

for categorie in categorie_ids.values():
    categorie_folder = os.path.join(path_to_save_save_converted_annotations, categorie)
    if not pathlib.Path(categorie_folder).is_dir():
        os.mkdir(categorie_folder)
    categorie_original_folder = os.path.join(path_to_save_save_converted_annotations, categorie, 'original')
    if not pathlib.Path(categorie_original_folder).is_dir():
        os.mkdir(categorie_original_folder)
    categorie_mask_folder = os.path.join(path_to_save_save_converted_annotations, categorie, 'masks')
    if not pathlib.Path(categorie_mask_folder).is_dir():
        os.mkdir(categorie_mask_folder)

counter = 0

for i in range(0, len(images)):
    image_name = images[i]
    print(i, '/', len(images), ' image name: ', image_name)

    image = Image.open(os.path.join(path_to_train_images, image_name + '.jpg'))
    image_mask_objects = Image.open(os.path.join(path_to_segmentation_masks_object_specific, image_name + '.png'))
    image_mask_classes = Image.open(os.path.join(path_to_segmentation_masks_class_specific, image_name + '.png'))
    image_mask_objects = np.asarray(image_mask_objects, dtype=np.uint8)
    image_mask_classes = np.asarray(image_mask_classes, dtype=np.uint8)

    unique_pixel_values_objects = np.unique(image_mask_objects)
    unique_pixel_values_objects = list(unique_pixel_values_objects)
    unique_pixel_values_classes = np.unique(image_mask_classes)
    unique_pixel_values_classes = list(unique_pixel_values_classes)

    if 255 in unique_pixel_values_objects:
        unique_pixel_values_objects.remove(255)
    if 255 in unique_pixel_values_classes:
        unique_pixel_values_classes.remove(255)
    if 0 in unique_pixel_values_objects:
        unique_pixel_values_objects.remove(0)
    if 0 in unique_pixel_values_classes:
        unique_pixel_values_classes.remove(0)

    for c in unique_pixel_values_objects:
        counter += 1
        image_mask_object_dependent = np.where(image_mask_objects == c, 255, 0)
        image_mask_object_dependent = np.asarray(image_mask_object_dependent, dtype=np.uint8)

        for l in unique_pixel_values_classes:
            image_mask_class_dependent = np.where(image_mask_classes == l, 255, 0)
            image_mask_class_dependent = np.asarray(image_mask_class_dependent, dtype=np.uint8)

            # cv2.imshow('object', image_mask_object_dependent)
            # cv2.imshow('class', image_mask_class_dependent)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            test = image_mask_object_dependent - image_mask_class_dependent
            t = list(np.unique(test))

            if 255 not in t:
                right_categorie = categorie_ids[l]

                x_min = np.min(image_mask_object_dependent.nonzero()[0])
                x_max = np.max(image_mask_object_dependent.nonzero()[0])
                y_min = np.min(image_mask_object_dependent.nonzero()[1])
                y_max = np.max(image_mask_object_dependent.nonzero()[1])

                cv2.imwrite(os.path.join(path_to_save_save_converted_annotations, right_categorie, 'masks',
                                         image_name + '_x_min=' + str(x_min) + '_x_max=' + str(x_max) + '_y_min=' + str(y_min) + '_y_max=' + str(y_max) + '.png'),
                            image_mask_object_dependent)
                image.save(os.path.join(path_to_save_save_converted_annotations, right_categorie, 'original',
                                         image_name + '_x_min=' + str(x_min) + '_x_max=' + str(x_max) + '_y_min=' + str(y_min) + '_y_max=' + str(y_max) + '.png'))
                break








