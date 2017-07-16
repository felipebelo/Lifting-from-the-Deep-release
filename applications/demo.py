#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose
from lifting.utils import config

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')

IMAGE_FILE_PATH = PROJECT_PATH + '/data/images/test_image.png'

SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'


def main():
    image = _read_rgb_image_file(IMAGE_FILE_PATH)

    # create pose estimator
    pose_estimator = PoseEstimator(
        config, {'size': image.shape}, SESSION_PATH, PROB_MODEL_PATH)

    # load model and run evaluation on image
    pose_estimator.initialise()

    # estimation
    pose_2d, pose_3d, visibility = pose_estimator.estimate(image)

    # Show 2D and 3D poses
    _display_results(image, pose_2d, pose_3d, visibility)

    # close model
    pose_estimator.close()


def _read_rgb_image_file(image_file_path):
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb
    return image


def _display_results(image, pose_2d, pose_3d, visibility):
    """Plot 2D and 3D poses for each of the people in the image."""
    _plot_pose_2d(image, pose_2d, visibility)
    _plot_pose_3d(pose_3d)


def _plot_pose_3d(pose_3d):
    for single_pose in pose_3d:
        plot_pose(single_pose)
    plt.show()


def _plot_pose_2d(in_image, pose_2d, visibility):
    plt.figure()
    draw_limbs(in_image, pose_2d, visibility)
    plt.imshow(in_image)
    plt.axis('off')


if __name__ == '__main__':
    import sys
    sys.exit(main())
