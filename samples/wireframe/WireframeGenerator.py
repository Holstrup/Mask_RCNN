import os
from PIL import Image
import random
import cv2 as cv
import json
import numpy as np
import h5py


# Define Dirs
MASK_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../"))
DATA_DIR = os.path.abspath(os.path.join(MASK_DIR, "datasets/wireframe"))

BACKGROUND_DIR = os.getcwd() + "/Backgrounds"
ICONS = os.listdir(os.getcwd() + "/Icons")
BACKGROUNDS = os.listdir(BACKGROUND_DIR)

# Wire-frame dimensions (iPhone 8)
WIDTH = 900
HEIGHT = 1200
DIMENSIONS = (WIDTH, HEIGHT)
bg_w, bg_h = DIMENSIONS
COLOR = (255, 255, 255, 255)

# Icon Dimensions
ICON_W = ICON_H = 40

#Specify training data or validation data
TYPES = ["/train", "/val"]


def get_mask(FILENAME):
    img = cv.imread(FILENAME, 0)
    _, th1 = cv.threshold(img,100,1,cv.THRESH_BINARY_INV)
    return th1

def remove_ds_file(dir_list):
    if ".DS_Store" in dir_list:
        dir_list.remove(".DS_Store")
    return dir_list


# Remove annoying Mac files
remove_ds_file(ICONS)
remove_ds_file(BACKGROUNDS)


def Save_image_infos(all_masks, all_class_names, type, NUM_IMAGES, ICONS_PER_IMAGE):
    data_to_write = np.random.random(size=(NUM_IMAGES, ICONS_PER_IMAGE, HEIGHT, WIDTH))

    with h5py.File(DATA_DIR + type + "/" + 'masks.h5', 'w') as hf:
        hf.create_dataset("data", data=all_masks)

    with open(DATA_DIR + type + "/" + "classes.txt", 'w') as f:
        for item in all_class_names:
            f.write("%s\n" % item)


def white_to_transparency(img):
    image_array = np.asarray(img.convert('RGBA')).copy()
    HEIGHT = np.shape(image_array)[0]
    INFLECTION_POINT = int(np.sum(image_array[0,:,0]) / HEIGHT) - 10
    INFLECTION_POINT = 100
    image_array[image_array > INFLECTION_POINT] = 255
    image_array[image_array < INFLECTION_POINT] = 0
    image_array[:, :, 3] = (255 * (image_array[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
    return Image.fromarray(image_array)


def generate_data_3(NUM_IMAGES, ICONS_PER_IMAGE):
    """
    Generates wireframes by inserting icons into backgrounds, and writing a "regions data" file
    that lets Mask R-CNN know where the BBoxes are on the image
    :param NUM_IMAGES: Number of images you want to generate
    :param ICONS_PER_IMAGE: MAX icons per image
    """
    for type in TYPES:
        if type == "/val":
            NUM_IMAGES = int(NUM_IMAGES / 5)

        all_masks = np.zeros((NUM_IMAGES, ICONS_PER_IMAGE, HEIGHT, WIDTH), dtype=np.bool)

        all_class_names = []
        for j in range(NUM_IMAGES):
            NUM_ICONS = random.randint(1, ICONS_PER_IMAGE) # Choose no. of icons in exact image

            #Pick background image
            background = Image.open(BACKGROUND_DIR + "/" + BACKGROUNDS[random.randint(0, len(BACKGROUNDS) - 1)]).convert("L")

            #Define total image mask - Just zeros
            image_mask = np.zeros((ICONS_PER_IMAGE, HEIGHT, WIDTH))


            #Class names - Tells us what icons are on the image
            class_names = []

            for i in range(NUM_ICONS):
                # Pick a random icon in the list of icons
                cur_icon = ICONS[random.randint(0, len(ICONS) - 1)]

                # Define directory path to that icon
                icons_in_dir = remove_ds_file(os.listdir('Icons/' + cur_icon + "/"))

                # Randomly pick an image file in the chosen directory
                exact_file = random.choice(icons_in_dir)

                # Open image file
                img = Image.open('Icons/' + cur_icon + "/" + exact_file, 'r')

                # Pick randomly where the icon should be placed
                offset = random.randint(1, bg_w - ICON_W), random.randint(1, bg_h - ICON_H)

                # Paste the icon in
                img = white_to_transparency(img)

                background.paste(img, offset, img)

                # Retrieve the mask of the icon file
                mask = get_mask('Icons/' + cur_icon + "/" + exact_file)

                # Insert mask into the same place as the icon was placed - in the image_mask variable
                image_mask[i, offset[1]:offset[1] + ICON_H, offset[0]: offset[0] + ICON_W] = mask


                # Add icon label to class names
                class_names.append((i, cur_icon))

            # File name is just an integer between 0 and NUM_IMAGES
            img_dir_name = DATA_DIR + type + '/' + str(j) + ".png"

            # Save background
            background.save(img_dir_name)

            all_masks[j, :, :, :] = image_mask

            all_class_names.append(class_names)


        Save_image_infos(all_masks, all_class_names, type, NUM_IMAGES, ICONS_PER_IMAGE)