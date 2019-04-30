import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import h5py
from matplotlib import pyplot as plt
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
    NUM_CLASSES = 1 + 10  # Background + objects

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class WireframeDataset(utils.Dataset):
    def load_wireframe(self, dataset_dir, subset, hc=False):
        """Load the surgery dataset from VIA.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val or predict
        """

        icon_dir = os.path.join(os.getcwd(), "Icons")
        icons = os.listdir(icon_dir)
        if ".DS_Store" in icons:
            icons.remove(".DS_Store")
        for i, icon in enumerate(icons):
            icon_name = icon.split(".")[0]
            self.add_class("wireframe", i + 1, icon_name)

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
            mask_path = dataset_dir + "/masks.h5"
            class_path = dataset_dir + "/classes.txt"

            self.add_image(
                source="wireframe",
                image_id=image_name,  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                mask_file=mask_path,
                class_file=class_path)


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
        with h5py.File(mask_file_path, 'r') as hf:
            mask = hf['data'][image_id, :, :, :]


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
        print(np.shape(mask))
        return mask.astype(np.bool), class_ids


    def load_mask_2(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a surgery dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "wireframe":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        class_names = info["names"]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
        # Assign class_ids by reading class_names
        class_ids = np.zeros([len(info["polygons"])])
        # In the surgery dataset, pictures are labeled with name 'a' and 'r' representing arm and ring.
        for i, p in enumerate(class_names):
            icon = list(filter(lambda icon: icon['name'] == p['name'], self.class_info))
            class_ids[i] = int(icon[0]["id"])
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
