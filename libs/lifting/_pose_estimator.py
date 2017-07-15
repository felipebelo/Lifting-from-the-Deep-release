# -*- coding: utf-8 -*-
"""
Created on Jul 13 16:20 2017

@author: Denis Tome'
"""
from lifting import utils
from lifting.utils import Process

import cv2
import numpy as np
import tensorflow as tf

import abc
ABC = abc.ABCMeta('ABC', (object,), {})

__all__ = [
    'PoseEstimatorInterface',
    'PoseEstimator'
]

UINT8_RANGE = [0, 255]


class PoseEstimatorInterface(ABC):

    @abc.abstractmethod
    def initialise(self):
        pass

    @abc.abstractmethod
    def estimate(self, image):
        return

    @abc.abstractmethod
    def close(self):
        pass


class PoseEstimator(PoseEstimatorInterface):

    def __init__(self, config, image_size, session_path, prob_model_path):
        """
        Initialising the graph in tensorflow.

        INPUT: image_size: Size of the image in the format (w x h x 3)
        """

        self.config = config
        self.session_path = session_path
        self.prob_model_path = prob_model_path

        original_image_height, original_image_width = (
            self.get_height_and_width(image_size))

        self.scale = self.compute_scale(original_image_height)
        self.image_width = int(self.scale * original_image_width)

        self.session = None

        self.image_in = None
        self.heatmap_person_large = None
        self.pose_image_in = None
        self.pose_centermap_in = None
        self.heatmap_pose = None

        self.initialised = False
        self.process = Process(config)

    def compute_scale(self, original_image_height):
        return self.config.INPUT_SIZE / float(original_image_height)

    @staticmethod
    def get_height_and_width(image_size):
        original_image_size = np.array(image_size)
        original_image_height = original_image_size[0]
        original_image_width = original_image_size[1]
        return original_image_height, original_image_width

    def initialise(self):
        """
        Load saved model in the graph

        INPUT: sess_path: path to the dir containing the
        tensorflow saved session

        OUTPUT: sess: tensorflow session
        """
        if self.initialised:
            return

        self._prepare_placeholders()
        self._load_model()

    def _load_model(self):
        session = tf.Session()
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, self.session_path)

        self.session = session

    def _prepare_placeholders(self):
        tf.reset_default_graph()

        with tf.variable_scope('CPM'):
            # placeholders for person network

            self.image_in = tf.placeholder(
                tf.float32,
                [1, self.config.INPUT_SIZE, self.image_width, 3]
            )

            heatmap_person = utils.inference_person(self.image_in)

            self.heatmap_person_large = tf.image.resize_images(
                heatmap_person,
                [self.config.INPUT_SIZE, self.image_width]
            )

            num = 16

            # placeholders for pose network
            self.pose_image_in = tf.placeholder(
                tf.float32,
                [num, self.config.INPUT_SIZE, self.config.INPUT_SIZE, 3]
            )

            self.pose_centermap_in = tf.placeholder(
                tf.float32,
                [num, self.config.INPUT_SIZE, self.config.INPUT_SIZE, 1]
            )

            self.heatmap_pose = utils.inference_pose(
                self.pose_image_in, self.pose_centermap_in)

    def estimate(self, image):
        """
        Estimate 2d and 3d poses on the image.

        INPUT:
            image: RGB image in the format (w x h x 3), UINT8
            sess: tensorflow session

        OUTPUT:
            pose_2d: 2D pose for each of the people in the image in the format
            (num_ppl x num_joints x 2) visibility: vector containing a bool
            value for each joint representing the visibility of the joint in
            the image (could be due to occlusions or the joint is not in the
            image) pose_3d: 3D pose for each of the people in the image in the
            format (num_ppl x 3 x num_joints)
        """
        if not self.initialised:
            self.initialise()

        resized_image = self._resize_image(image)
        b_image = self._prepare_b_image(resized_image)

        centers = self._compute_centers(b_image)
        hmap_pose = self._compute_hmap_pose(resized_image, b_image, centers)

        # Estimate 2D poses
        pose_2d_raw, visibility = self._estimate_2d(hmap_pose, centers)

        # Estimate 3D poses
        pose_3d = self._estimate_3d(pose_2d_raw, visibility)

        pose_2d = self._prepare_pose_2d(pose_2d_raw)
        return pose_2d, pose_3d, visibility

    def _resize_image(self, image):
        return cv2.resize(image,
                          None,
                          fx=self.scale, fy=self.scale,
                          interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _prepare_b_image(image):
        normalised_image = image[np.newaxis] / float(UINT8_RANGE[1])

        # noinspection PyTypeChecker
        return np.array(
            normalised_image - 0.5,
            dtype=np.float32
        )

    def _compute_centers(self, b_image):
        hmap_person = self.session.run(
            self.heatmap_person_large,
            {self.image_in: b_image}
        )
        hmap_person = np.squeeze(hmap_person)
        centers = self.process.detect_objects_heatmap(hmap_person)
        return centers

    def _compute_hmap_pose(self, image, b_image, centers):
        image_width = image.shape[1]

        b_pose_image, b_pose_cmap = self.process.prepare_input_posenet(
            b_image[0], centers,
            [self.config.INPUT_SIZE, image_width],
            [self.config.INPUT_SIZE, self.config.INPUT_SIZE])

        feed_dict = {
            self.pose_image_in: b_pose_image,
            self.pose_centermap_in: b_pose_cmap
        }

        hmap_pose = self.session.run(self.heatmap_pose, feed_dict)
        return hmap_pose

    def _prepare_pose_2d(self, pose_2d_raw):
        pose_2d_normalised = np.round(pose_2d_raw / self.scale)  # Normalise
        pose_2d = pose_2d_normalised.astype(np.int32)  # Convert type
        return pose_2d

    def _estimate_2d(self, hmap_pose, centers):
        pose_2d_raw, visibility = (
            self.process.detect_parts_heatmaps(
                hmap_pose, centers,
                [
                    self.config.INPUT_SIZE,
                    self.config.INPUT_SIZE
                ]
            ))

        return pose_2d_raw, visibility

    def _estimate_3d(self, pose_2d_raw, visibility):
        pose_lifting = utils.Prob3dPose(self.prob_model_path)

        transformed_pose2d, weights = (
            pose_lifting.transform_joints(
                pose_2d_raw.copy(), visibility)
        )
        pose_3d = pose_lifting.compute_3d(transformed_pose2d, weights)
        return pose_3d

    def close(self):
        self.session.close()
