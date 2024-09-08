# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------

# general package imports
import os
import sys
import numpy as np
import torch
from easydict import EasyDict as edict

# add project directory to python path to enable relative imports
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related imports
from tools.objdet_models.resnet.models import fpn_resnet
from tools.objdet_models.resnet.utils.evaluation_utils import decode, post_processing
from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2


def load_configs_model(model_name='darknet', configs=None):
    """
    Load model-related parameters into an EasyDict (configs).
    """

    # Initialize config file if none has been passed
    if configs is None:
        configs = edict()

    # Get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = configs.model_path = os.path.abspath(os.path.join(curr_path, os.pardir))

    # Set parameters according to the model type
    if model_name == "darknet":
        configs.model_path = os.path.join(parent_path, 'tools', 'objdet_models', 'darknet')
        configs.pretrained_filename = os.path.join(configs.model_path, 'pretrained', 'complex_yolov4_mse_loss.pth')
        configs.arch = 'darknet'
        configs.batch_size = 4
        configs.cfgfile = os.path.join(configs.model_path, 'config', 'complex_yolov4.cfg')
        configs.conf_thresh = 0.5
        configs.distributed = False
        configs.img_size = 608
        configs.nms_thresh = 0.4
        configs.num_samples = None
        configs.num_workers = 4
        configs.pin_memory = True
        configs.use_giou_loss = False

    elif model_name == 'fpn_resnet':
        ####### ID_S3_EX1-3 START #######
        print("student task ID_S3_EX1-3")
        ####### ID_S3_EX1-3 END #######

    else:
        raise ValueError("Error: Invalid model name")

    # GPU vs. CPU
    configs.no_cuda = True  # if True, CUDA is not used
    configs.gpu_idx = 0  # GPU index to use
    configs.device = torch.device('cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}')

    return configs


def load_configs(model_name='fpn_resnet', configs=None):
    """
    Load all object-detection parameters into an EasyDict (configs).
    """

    # Initialize config file if none has been passed
    if configs is None:
        configs = edict()

    # Birds-eye view (BEV) parameters
    configs.lim_x = [0, 50]  # Detection range in meters
    configs.lim_y = [-25, 25]
    configs.lim_z = [-1, 3]
    configs.lim_r = [0, 1.0]  # Reflected lidar intensity
    configs.bev_width = 608  # Pixel resolution of BEV image
    configs.bev_height = 608

    # Add model-dependent parameters
    configs = load_configs_model(model_name, configs)

    # Visualization parameters
    configs.output_width = 608  # Width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]]  # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2

    return configs


def create_model(configs):
    """
    Create a model according to the selected model type.
    """

    # Check for availability of the model file
    assert os.path.isfile(configs.pretrained_filename), f"No file at {configs.pretrained_filename}"

    # Create model depending on the architecture name
    if configs.arch == 'darknet' and configs.cfgfile is not None:
        print('Using darknet')
        model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)

    elif 'fpn_resnet' in configs.arch:
        print('Using ResNet architecture with feature pyramid')

        ####### ID_S3_EX1-4 START #######
        print("student task ID_S3_EX1-4")
        ####### ID_S3_EX1-4 END #######

    else:
        assert False, 'Undefined model backbone'

    # Load model weights
    model.load_state_dict(torch.load(configs.pretrained_filename, map_location='cpu'))
    print(f'Loaded weights from {configs.pretrained_filename}\n')

    # Set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else f'cuda:{configs.gpu_idx}')
    model = model.to(device=configs.device)  # Load model to either CPU or GPU
    model.eval()

    return model


def detect_objects(input_bev_maps, model, configs):
    """
    Detect trained objects in birds-eye view (BEV) using the selected model.
    """

    # Deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():

        # Perform inference
        outputs = model(input_bev_maps)

        # Decode model output into target object format
        if 'darknet' in configs.arch:
            # Perform post-processing
            output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            detections = []
            for sample_i in range(len(output_post)):
                if output_post[sample_i] is None:
                    continue
                detection = output_post[sample_i]
                for obj in detection:
                    x, y, w, l, im, re, _, _, _ = obj
                    yaw = np.arctan2(im, re)
                    detections.append([1, x, y, 0.0, 1.50, w, l, yaw])

        elif 'fpn_resnet' in configs.arch:
            # Decode output and perform post-processing

            ####### ID_S3_EX1-5 START #######
            print("student task ID_S3_EX1-5")
            ####### ID_S3_EX1-5 END #######

    ####### ID_S3_EX2 START #######
    # Extract 3D bounding boxes from model response
    print("student task ID_S3_EX2")
    objects = []

    # Step 1: Check whether there are any detections
    if len(detections) > 0:

        # Step 2: Loop over all detections
        for det in detections:

            # Step 3: Perform the conversion using the limits for x, y, and z set in the configs structure
            _, x, y, z, h, w, l, yaw = det

            # Apply scaling factors to ensure the bounding box aligns with BEV image limits
            x = x / configs.bev_width * (configs.lim_x[1] - configs.lim_x[0]) + configs.lim_x[0]
            y = y / configs.bev_height * (configs.lim_y[1] - configs.lim_y[0]) + configs.lim_y[0]

            # Set object position and dimensions
            obj = [x, y, z, w, l, h, yaw]

            # Step 4: Append the current object to the 'objects' array
            objects.append(obj)

    ####### ID_S3_EX2 END #######

    return objects
