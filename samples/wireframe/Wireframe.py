import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import h5py
from matplotlib import pyplot as plt
from PIL import Image
import random
import cv2 as cv


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

import keras
keras.layers.TimeDistributed(keras.layers.Flatten())


# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

from mrcnn import visualize
# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
from PIL import Image
# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class WireframeConfig(Config):
    """Configuration for training on the toy dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "wireframe"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 15  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 200

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7




############################################################
#  Dataset
############################################################

class WireframeDataset(utils.Dataset):

    def __init__(self):
        self._image_ids = []
        self.image_info = []
        # Background is always the first class
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

        # Define Dirs
        self.MASK_DIR = os.path.abspath(os.path.join(os.getcwd(), "../../"))
        self.DATA_DIR = os.path.abspath(os.path.join(self.MASK_DIR, "datasets/wireframe"))

        self.BACKGROUND_DIR = os.getcwd() + "/Backgrounds"
        self.ICONS = os.listdir(os.getcwd() + "/Icons")
        self.BACKGROUNDS = os.listdir(self.BACKGROUND_DIR)

        # Wire-frame dimensions (iPhone 8)
        self.WIDTH = 900
        self.HEIGHT = 1200
        self.DIMENSIONS = (self.WIDTH,self.HEIGHT)
        self.bg_w, self.bg_h = self.DIMENSIONS

        # Icon Dimensions
        self.ICON_W = self.ICON_H = 40

        # Specify training data or validation data
        self.TYPES = ["/train", "/val"]

        # Remove annoying Mac files
        self.remove_ds_file(self.ICONS)
        self.remove_ds_file(self.BACKGROUNDS)

    def get_mask(self, FILENAME):
        img = cv.imread(FILENAME, 0)
        _, th1 = cv.threshold(img, 100, 1, cv.THRESH_BINARY_INV)
        return th1

    def remove_ds_file(self, dir_list):
        if ".DS_Store" in dir_list:
            dir_list.remove(".DS_Store")
        return dir_list

    def Save_image_infos(self, all_class_names, type):
        with open(self.DATA_DIR + type + "/" + "classes.txt", 'w') as f:
            for item in all_class_names:
                f.write("%s\n" % item)

    def save_mask(self, masks, type, image_n):
        np.save(self.DATA_DIR + type + "/" + str(image_n), masks)

    def white_to_transparency(self, img):
        image_array = np.asarray(img.convert('RGBA')).copy()
        HEIGHT = np.shape(image_array)[0]
        INFLECTION_POINT = int(np.sum(image_array[0, :, 0]) / HEIGHT) - 10
        INFLECTION_POINT = 100
        image_array[image_array > INFLECTION_POINT] = 255
        image_array[image_array < INFLECTION_POINT] = 0
        image_array[:, :, 3] = (255 * (image_array[:, :, :3] != 255).any(axis=2)).astype(np.uint8)
        return Image.fromarray(image_array)


    def load_wireframe(self, dataset_dir, subset, hc=False):
        """Load the surgery dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """

        icon_dir = os.path.join(os.getcwd(), "Icons")
        icons = os.listdir(icon_dir)
        if ".DS_Store" in icons:
            icons.remove(".DS_Store")

        self.add_class("wireframe", 1, "A")
        self.add_class("wireframe", 2, "Cross")
        self.add_class("wireframe", 3, "D")
        self.add_class("wireframe", 4, "Done")
        self.add_class("wireframe", 5, "H")
        self.add_class("wireframe", 6, "Heart")
        self.add_class("wireframe", 7, "Home")
        self.add_class("wireframe", 8, "I")
        self.add_class("wireframe", 9, "Menu")
        self.add_class("wireframe", 10, "More")
        self.add_class("wireframe", 11, "R")
        self.add_class("wireframe", 12, "Search")
        self.add_class("wireframe", 13, "U")
        self.add_class("wireframe", 14, "Wifi")
        self.add_class("wireframe", 15, "Z")


        # Train or validation dataset?
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)
        image_files = [fi for fi in os.listdir(dataset_dir) if fi.endswith(".png")]

        # Add images
        for i in range(len(image_files)):
            image_name = str(i) + ".png"
            image_path = os.path.join(dataset_dir, image_name)
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]
            mask_path = dataset_dir + "/" + str(i) + ".npy"
            class_path = dataset_dir + "/classes.txt"

            self.add_image(
                source="wireframe",
                image_id=image_name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask_file=mask_path,
                class_file=class_path)



    def generate_data(self, NUM_IMAGES, ICONS_PER_IMAGE):
        """
        Generates wireframes by inserting icons into backgrounds, and writing a "regions data" file
        that lets Mask R-CNN know where the BBoxes are on the image
        :param NUM_IMAGES: Number of images you want to generate
        :param ICONS_PER_IMAGE: MAX icons per image
        """
        for type in self.TYPES:
            if type == "/val":
                NUM_IMAGES = int(NUM_IMAGES / 5)

            all_class_names = []
            for j in range(NUM_IMAGES):
                NUM_ICONS = random.randint(1, ICONS_PER_IMAGE)  # Choose no. of icons in exact image

                # Pick background image
                background = Image.open(
                    self.BACKGROUND_DIR + "/" + self.BACKGROUNDS[random.randint(0,
                                                                                len(self.BACKGROUNDS) - 1)]).convert("L")

                # Define total image mask - Just zeros
                image_mask = np.zeros((ICONS_PER_IMAGE, self.HEIGHT, self.WIDTH), dtype=np.bool)

                # Class names - Tells us what icons are on the image
                class_names = []

                for i in range(NUM_ICONS):
                    # Pick a random icon in the list of icons
                    cur_icon = self.ICONS[random.randint(0, len(self.ICONS) - 1)]

                    # Define directory path to that icon
                    icons_in_dir = self.remove_ds_file(os.listdir('Icons/' + cur_icon + "/"))

                    # Randomly pick an image file in the chosen directory
                    exact_file = random.choice(icons_in_dir)

                    # Open image file
                    img = Image.open('Icons/' + cur_icon + "/" + exact_file, 'r')

                    # Pick randomly where the icon should be placed
                    offset = random.randint(1, self.bg_w - self.ICON_W), random.randint(1, self.bg_h - self.ICON_H)

                    # Paste the icon in
                    img = self.white_to_transparency(img)

                    background.paste(img, offset, img)

                    # Retrieve the mask of the icon file
                    mask = self.get_mask('Icons/' + cur_icon + "/" + exact_file)

                    # Insert mask into the same place as the icon was placed - in the image_mask variable
                    image_mask[i, offset[1]:offset[1] + self.ICON_H, offset[0]: offset[0] + self.ICON_W] = mask

                    # Add icon label to class names
                    class_names.append((i, cur_icon))

                # File name is just an integer between 0 and NUM_IMAGES
                img_dir_name = self.DATA_DIR + type + '/' + str(j) + ".png"

                # Save background
                background.save(img_dir_name)

                self.save_mask(image_mask, type, j)

                all_class_names.append(class_names)

            self.Save_image_infos(all_class_names, type)


    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        # Check if image belongs to a wireframe dataset (which it always will)
        image_info = self.image_info[image_id]
        if image_info["source"] != "wireframe":
            return super(self.__class__, self).load_mask(image_id)


        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask_file_path = info["mask_file"]

        class_names = []
        with open(info["class_file"]) as f:
            while 1:
                line = f.readline()
                if not line: break
                class_names.append(eval(line))
            f.close()

        # Load mask file
        mask = np.load(mask_file_path)
        mask = np.swapaxes(mask, 0, 2)
        mask = np.swapaxes(mask, 0, 1)


        # Assign class_ids by reading class_names
        class_ids = np.zeros(len(class_names[image_id]))
        # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        #print(class_names[image_id])

        for i, p in enumerate(class_names[image_id]):
            icon = list(filter(lambda icon: icon['name'] == p[1], self.class_info))
            class_ids[i] = int(icon[0]["id"])

        mask = mask[:, :, 0:len(class_ids)]


        class_ids = class_ids.astype(int)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), class_ids



    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "wireframe":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


    def load_natural_image(self, dataset_dir, subset):
        """Load the surgery dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """
        icon_dir = os.path.join(os.getcwd(), "Icons")
        icons = os.listdir(icon_dir)
        for i, icon in enumerate(icons):
            icon_name = icon.split(".")[0]
            self.add_class("wireframe", i + 1, icon_name)
        print(self.class_info)
        # Prediction data set?
        assert subset in ["predict"]
        dataset_dir = os.path.join(dataset_dir, subset)

        # Add images
        for image_file in os.listdir(dataset_dir):
            if image_file == "__init__.py" or image_file == ".DS_Store":
                continue
            else:
                image_path = os.path.join(dataset_dir, image_file)
                image = skimage.io.imread(image_path)
                height, width = image.shape[:2]
                id = int(image_file.split(".")[0])
                self.image_ids.append(id)
                self.add_image(
                    "wireframe",
                    image_id=id,
                    path=image_path,
                    width=width, height=height,
                    polygons="",
                    names=""
                )
