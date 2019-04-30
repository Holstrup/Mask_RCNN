import os
from PIL import Image
import random
import cv2 as cv
import json
import numpy as np


BACKGROUND_DIR = os.getcwd() + "/Backgrounds"
ICONS = os.listdir(os.getcwd() + "/Icons")
BACKGROUNDS = os.listdir(BACKGROUND_DIR)

# Remove annoying Mac files
if ".DS_Store" in ICONS:
    ICONS.remove(".DS_Store")
if ".DS_Store" in BACKGROUNDS:
    BACKGROUNDS.remove(".DS_Store")

# Define Dirs
MASK_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../"))
DATA_DIR = os.path.abspath(os.path.join(MASK_DIR, "datasets/wireframe"))

# Wire-frame dimensions (iPhone 8)
WIDTH = 900
HEIGHT = 1200
DIMENSIONS = (WIDTH, HEIGHT)
bg_w, bg_h = DIMENSIONS
COLOR = (255, 255, 255, 255)

# Icon Dimensions
ICON_W = ICON_H = 100

#Specify training data or validation data
TYPES = ["/train", "/val"]


file_content = "{"


def write_string_to_json(string, type):
    """
    :param string: Content to write to file
    :param type: Train or validation dir
    """
    with open(DATA_DIR + type + '/via_region_data.json', 'w+') as f:
        f.write(string)


def add_box_json_polygon(filename, size, icons):
    """
    Helper function for the generate data function

    :param filename: Filename or image file
    :param size: Size of the image
    :param icons: List of icons in image
    :return:
    """
    s = " '{}{}':{{ 'filename': '{}', 'size': {}, 'regions': [".format(filename, size, filename, size)
    for icon in icons:
        x1, x2 = (icon[1], icon[1] + ICON_W)
        y1, y2 = (icon[2], icon[2] + ICON_H)
        s += "{{'shape_attributes': {{'name': 'polygon', " \
             "'all_points_x': [{},{},{},{}], 'all_points_y': [{},{},{},{}] }}," \
             "'region_attributes': {{'name': '{}'}} }} ,".format(x1, x1, x2, x2, y2, y1, y1, y2, icon[0])
    s = s[0:-1] + "]" + " },"
    s = s.replace("'", '"')
    return s

def get_mask(FILENAME):
    img = cv.imread(FILENAME, 0)
    _, th1 = cv.threshold(img,100,1,cv.THRESH_BINARY_INV)
    return th1

def remove_ds_file(dir_list):
    if ".DS_Store" in dir_list:
        dir_list.remove(".DS_Store")
    return dir_list

def Save_image_infos(all_masks, all_class_names, type):
    np.save(DATA_DIR + type + "/" + "masks", all_masks)
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

        all_masks = np.zeros((NUM_IMAGES, ICONS_PER_IMAGE, HEIGHT, WIDTH))
        #all_masks = np.zeros((NUM_IMAGES, ICONS_PER_IMAGE, WIDTH, HEIGHT))
        all_class_names = []
        for j in range(NUM_IMAGES):
            NUM_ICONS = random.randint(1, ICONS_PER_IMAGE) # Choose no. of icons in exact image

            #Pick background image
            background = Image.open(BACKGROUND_DIR + "/" + BACKGROUNDS[random.randint(0, len(BACKGROUNDS) - 1)]).convert("L")

            #Define total image mask - Just zeros
            image_mask = np.zeros((ICONS_PER_IMAGE, HEIGHT, WIDTH))
            #image_mask = np.zeros((ICONS_PER_IMAGE, WIDTH, HEIGHT))

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

                img = Image.fromarray(255 * image_mask[0, :, :])


                # Add icon label to class names
                class_names.append((i, cur_icon))

            # File name is just an integer between 0 and NUM_IMAGES
            img_dir_name = DATA_DIR + type + '/' + str(j) + ".png"

            # Save background
            background.save(img_dir_name)

            all_masks[j, :, :, :] = image_mask

            all_class_names.append(class_names)

        Save_image_infos(all_masks, all_class_names, type)


def generate_data_2(NUM_IMAGES, ICONS_PER_IMAGE):
    """
    Generates wireframes by inserting icons into backgrounds, and writing a "regions data" file
    that lets Mask R-CNN know where the BBoxes are on the image

    :param NUM_IMAGES: Number of images you want to generate
    :param ICONS_PER_IMAGE: MAX icons per image
    """
    global file_content
    for type in TYPES:
        file_content = "{"
        if type == "/val":
            NUM_IMAGES = int(NUM_IMAGES / 5)
        for j in range(NUM_IMAGES):
            icon_list = []
            NUM_ICONS = random.randint(1, ICONS_PER_IMAGE)
            background = Image.open(BACKGROUND_DIR + "/" + BACKGROUNDS[random.randint(0, len(BACKGROUNDS) - 1)]).convert("L")
            for i in range(NUM_ICONS):
                cur_icon = ICONS[random.randint(0, len(ICONS) - 1)]
                icons_in_dir = os.listdir('Icons/' + cur_icon + "/")
                if ".DS_Store" in icons_in_dir:
                    icons_in_dir.remove(".DS_Store")
                exact_file = random.choice(icons_in_dir)
                img = Image.open('Icons/' + cur_icon + "/" + exact_file, 'r').resize((ICON_W, ICON_H))
                offset = random.randint(1, bg_w - ICON_W), random.randint(1, bg_h - ICON_H)
                background.paste(img, offset, img)
                icon_list.append((cur_icon, offset[0], offset[1]))
            img_dir_name = DATA_DIR + type + '/' + str(j) + ".png"
            background.save(img_dir_name)
            filename = str(j) + ".png"
            img_size = os.stat(img_dir_name).st_size
            file_content += add_box_json_polygon(filename, img_size, icon_list)

        file_content = file_content[0:-1] + "}"
        write_string_to_json(file_content, type)

def generate_data(NUM_IMAGES, ICONS_PER_IMAGE):
    """
    Generates wireframes by inserting icons into backgrounds, and writing a "regions data" file
    that lets Mask R-CNN know where the BBoxes are on the image

    :param NUM_IMAGES: Number of images you want to generate
    :param ICONS_PER_IMAGE: MAX icons per image
    """
    global file_content
    for type in TYPES:
        file_content = "{"
        if type == "/val":
            NUM_IMAGES = int(NUM_IMAGES / 5)
        for j in range(NUM_IMAGES):
            icon_list = []
            NUM_ICONS = random.randint(1, ICONS_PER_IMAGE)
            background = Image.open(BACKGROUND_DIR + "/" + BACKGROUNDS[random.randint(0, len(BACKGROUNDS) - 1)]).convert("L")
            for i in range(NUM_ICONS):
                cur_icon = ICONS[random.randint(0, len(ICONS) - 1)]
                img = Image.open('Icons/' + cur_icon, 'r').resize((ICON_W, ICON_H))
                offset = random.randint(1, bg_w - ICON_W), random.randint(1, bg_h - ICON_H)
                background.paste(img, offset, img)
                icon_list.append((cur_icon[0:-4], offset[0], offset[1]))
            img_dir_name = DATA_DIR + type + '/' + str(j) + ".png"
            background.save(img_dir_name)
            filename = str(j) + ".png"
            img_size = os.stat(img_dir_name).st_size
            file_content += add_box_json_polygon(filename, img_size, icon_list)

        file_content = file_content[0:-1] + "}"
        write_string_to_json(file_content, type)
