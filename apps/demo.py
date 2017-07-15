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


def main():
    dir_path = dirname(realpath(__file__))
    project_path = realpath(dir_path + '/..')

    image_file_path = project_path + '/data/images/test_image.png'
    image = read_rgb_image_file(image_file_path)

    # create pose estimator
    saved_sessions_dir = project_path + '/data/saved_sessions'
    session_path = saved_sessions_dir + '/init_session/init'
    prob_model_path = saved_sessions_dir + '/prob_model/prob_model_params.mat'

    image_size = image.shape

    pose_estimator = PoseEstimator(
        config, image_size, session_path, prob_model_path)

    # load model and run evaluation on image
    pose_estimator.initialise()

    # estimation
    pose_2d, pose_3d, visibility = pose_estimator.estimate(image)

    # close model
    pose_estimator.close()

    # Show 2D and 3D poses
    display_results(image, pose_2d, visibility, pose_3d)


def read_rgb_image_file(image_file_path):
    image = cv2.imread(image_file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # conversion to rgb
    return image


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        # or plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()


if __name__ == '__main__':
    import sys
    sys.exit(main())
