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

    def __init__(self, config, image_properties_dict,
                 session_path, prob_model_path):
        """
        Initialising the graph in tensorflow.

        INPUTS:
        image_properties:
        {
            'size': Size of the image in the format (w x h x 3),
            'range' : Values range, default is UINT8 - [0, 255]
        }

        sess_path: path to the dir containing the tensorflow saved
        session

        prob_model: path to the prob. model parameters file
        """

        image_properties = _ImageProperties(
            image_properties_dict, config)

        self._placeholders = _Placeholders(image_properties)
        self._session = _Session(session_path, self._placeholders)

        self._estimators = _Estimators(
            self._session, config, image_properties, prob_model_path)

    def initialise(self):
        """
        Load saved model in the graph
        """
        if self._is_initialised():
            return

        # enforce initialisation of placeholders
        self._placeholders.get_placeholders()

        # enforce initialisation of session
        self._session.get()

    def estimate(self, image):
        """
        Estimate 2d and 3d poses on the image.

        INPUT:
            image: RGB image in the format (w x h x 3)

        OUTPUT:
            pose_2d: 2D pose for each of the people in the image in the format
            (num_ppl x num_joints x 2)

            visibility: vector containing a bool
            value for each joint representing the visibility of the joint in
            the image (could be due to occlusions or the joint is not in the
            image)

            pose_3d: 3D pose for each of the people in the image in the
            format (num_ppl x 3 x num_joints)
        """
        self.initialise()   # will initialise only if needed

        self._estimators.use_image(image)

        pose_2d, visibility = self._estimators.get_2d_estimate()
        pose_3d = self._estimators.get_3d_estimate()

        return pose_2d, pose_3d, visibility

    def close(self):
        self._session.close()

    def _is_initialised(self):
        return self._session.has() and self._placeholders.has()


class _Estimators:

    def __init__(self, session, config, image_properties, prob_model_path):
        self._session = session
        self._process = Process(config)
        self._image_properties = image_properties
        self._prob_model_path = prob_model_path

        self._networks = None
        self._estimator_2d = None
        self._estimator_3d = None

    def use_image(self, image):
        self._networks = _PersonPoseNetworks(
            image, self._session, self._process, self._image_properties)

        self._estimator_2d = _Estimator2d(
            self._process, self._image_properties, self._networks)

        self._estimator_3d = _Estimator3d(
            self._estimator_2d, self._prob_model_path)

    def run_person_and_pose_networks(self):
        self._networks.run_person_and_pose_networks()

    def get_2d_estimate(self):
        return self._estimator_2d.get()

    def get_3d_estimate(self):
        return self._estimator_3d.get()


class _PersonPoseNetworks:
    def __init__(self, image, session, process, image_properties):
        self._image = image
        self._session = session
        self._process = process
        self._image_properties = image_properties

        self._pose_centermap = None
        self._pose_heatmap = None

    def get(self):
        self.run_person_and_pose_networks()
        return self._pose_heatmap, self._pose_centermap

    def run_person_and_pose_networks(self):
        if not self._has_run():
            person_b_image, person_heatmap = self._run_person_network()

            self._pose_centermap, self._pose_heatmap = (
                self._run_pose_network(person_b_image, person_heatmap))

    def _has_run(self):
        return (
            _exists(self._pose_centermap) and
            _exists(self._pose_heatmap))

    def _run_person_network(self):
        person_b_image = self._compute_person_b_in()
        person_heatmap = self._session.run_person_network(person_b_image)
        return person_b_image, person_heatmap

    def _run_pose_network(self, person_b_image, person_heatmap):
        pose_image, pose_centermap = (
            self._compute_pose_in(person_b_image, person_heatmap))

        pose_b_image, pose_b_centermap = (
            self._compute_pose_b_in(pose_image, pose_centermap))

        pose_heatmap = (
            self._session.run_pose_network(pose_b_image, pose_b_centermap))

        return pose_centermap, pose_heatmap

    def _compute_pose_in(self, person_b_image, person_heatmap):
        pose_image = person_b_image
        pose_centermap = self._compute_pose_centermap(person_heatmap)

        return pose_image, pose_centermap

    def _compute_pose_centermap(self, person_heatmap):
        person_heatmap = np.squeeze(person_heatmap)
        pose_centermap = self._process.detect_objects_heatmap(person_heatmap)
        return pose_centermap

    def _compute_pose_b_in(self, pose_image, pose_centermap):
        person_image_size = self._image_properties.get_scaled_size()
        pose_image_size = self._image_properties.get_squared_size()

        pose_b_image, pose_b_centermap = self._process.prepare_input_posenet(
            pose_image[0],
            pose_centermap,
            person_image_size,
            pose_image_size)

        return pose_b_image, pose_b_centermap

    def _compute_person_b_in(self):
        return _compute_b_image(
            self._image,
            self._image_properties.get_scale(),
            self._image_properties.get_range())


def _exists(attribute):
    return attribute is not None


def _compute_b_image(image, scale, image_range):
    resized_image = _resize_image(image, scale)

    b_image = resized_image[np.newaxis]
    b_image = b_image - float(image_range[0])
    b_image = b_image / float(image_range[1])
    b_image = b_image - 0.5
    b_image = np.array(b_image, dtype=np.float32)
    return b_image


def _resize_image(image, scale):
    return cv2.resize(
        image,
        None,
        fx=scale, fy=scale,
        interpolation=cv2.INTER_CUBIC)


class _Estimator2d:
    def __init__(self, process, image_properties, networks):
        self._process = process
        self._image_properties = image_properties
        self._networks = networks

        self._pose_2d = None
        self._raw_pose_2d = None
        self._visibility = None

    @property
    def raw_pose_2d(self):
        self.get()
        return self._raw_pose_2d

    def get(self):
        if not self.has():
            self._raw_pose_2d, self._visibility = self._compute()
            self._pose_2d = self._normalise(self._raw_pose_2d)

        return self._pose_2d, self._visibility

    def has(self):
        return (_exists(self._raw_pose_2d) and
                _exists(self._pose_2d) and
                _exists(self._visibility))

    def _compute(self):
        pose_heatmap, pose_centermap = self._networks.get()

        return (
            self._process.detect_parts_heatmaps(
                pose_heatmap,
                pose_centermap,
                self._image_properties.get_squared_size()))

    def _normalise(self, pose_2d_raw):
        return _normalise_pose_2d(
            pose_2d_raw,
            self._image_properties.get_scale())


def _normalise_pose_2d(pose_2d, scale):
    new_pose_2d = pose_2d.copy()
    new_pose_2d = new_pose_2d / scale  # normalising
    new_pose_2d = np.round(new_pose_2d)
    new_pose_2d = new_pose_2d.astype(np.int32)  # type conversion
    return new_pose_2d


class _Estimator3d:
    def __init__(self, estimator_2d, prob_model_path):
        self._estimator_2d = estimator_2d
        self._pose_lifting = utils.Prob3dPose(prob_model_path)

        self._pose_3d = None

    def get(self):
        if not self._has():
            self._pose_3d = self._compute()

        return self._pose_3d

    def _compute(self):
        pose_2d, visibility = self._estimator_2d.get()
        raw_pose_2d = self._estimator_2d.raw_pose_2d

        pose_2d_transformed, weights = (
            self._pose_lifting.transform_joints(
                raw_pose_2d, visibility))

        return self._pose_lifting.compute_3d(pose_2d_transformed, weights)

    def _has(self):
        return _exists(self._pose_3d)


class _ImageProperties:

    def __init__(self, properties_dict, config):
        self.properties_dict = properties_dict
        self._config = config

        self._scale = None
        self._scaled_size = None
        self._squared_size = None

    def get_range(self):
        return self.properties_dict.get('range', UINT8_RANGE)

    def get_size(self):
        return np.array(self.properties_dict.get('size', [0, 0]))

    def get_scale(self):  # lazy initialisation
        if not self._has_scale():
            self._scale = self._compute_scale()
        return self._scale

    def _has_scale(self):
        return _exists(self._scale)

    def _compute_scale(self):
        size = self.get_size()
        height = _get_height(size)
        new_height = self._config.INPUT_SIZE
        return new_height / float(height)

    def get_scaled_size(self):  # lazy initialisation
        if not self._has_scaled_size():
            self._scaled_size = self._compute_scaled_size()

        return self._scaled_size

    def _has_scaled_size(self):
        return _exists(self._scaled_size)

    def _compute_scaled_size(self):
        scale = self.get_scale()
        size = self.get_size()
        person_image_size = _rescale_size(scale, size)
        return person_image_size

    def get_squared_size(self):  # lazy initialisation
        if not self._has_squared_size():
            self._squared_size = self._compute_squared_size()

        return self._squared_size

    def _has_squared_size(self):
        return _exists(self._squared_size)

    def _compute_squared_size(self):
        person_image_size = self.get_scaled_size()
        height = _get_height(person_image_size)
        pose_image_size = [height, height]
        return pose_image_size


def _get_height(image_size):
    return image_size[0]


def _get_width(image_size):
    return image_size[1]


def _rescale_size(scale, size):
    width = int(scale * _get_width(size))
    height = int(scale * _get_height(size))
    return height, width


class _Session:

    def __init__(self, path, placeholders):
        self._path = path
        self._session = None
        self._placeholders = placeholders

    def get(self):  # lazy initialisation
        if not self.has():
            self._session = _load_model(self._path)

        return self._session

    def has(self):
        return _exists(self._session)

    def close(self):
        if self.has():
            self._session.close()
            self._session = None

    def run(self, *args, **kargs):
        return self.get().run(*args, **kargs)

    def run_person_network(self, b_image):
        person_heatmap = self._session.run(
            self._placeholders.get('person heatmap'),
            self._get_person_feed_dict(b_image)
        )

        return person_heatmap

    def _get_person_feed_dict(self, b_image):
        return {
            self._placeholders.get('person image in'): b_image
        }

    def run_pose_network(self, b_pose_image, b_pose_centermap):
        pose_heatmap = self._session.run(
            self._placeholders.get('pose heatmap'),
            self._get_pose_feed_dict(b_pose_centermap, b_pose_image))

        return pose_heatmap

    def _get_pose_feed_dict(self, b_pose_centermap, b_pose_image):
        pose_feed_dict = {
            self._placeholders.get('pose image in'): b_pose_image,
            self._placeholders.get('pose centermap in'): b_pose_centermap
        }
        return pose_feed_dict


def _load_model(path):
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(session, path)

    return session


class _Placeholders:
    POSE_DIM_0 = 16

    def __init__(self, image_properties):
        self._image_properties = image_properties
        self._placeholders = None

    def get(self, key):
        placeholders = self.get_placeholders()

        return placeholders.get(key, None)

    def get_placeholders(self):  # lazy initialisation
        if not self.has():
            self._placeholders = self._compute()

        return self._placeholders

    def has(self):
        return _exists(self._placeholders)

    def _compute(self):
        tf.reset_default_graph()

        with tf.variable_scope('CPM'):
            person_image_in, person_heatmap = (
                self._compute_person_network_placeholders())

            pose_image_in, pose_centermap_in, pose_heatmap_in = (
                self._compute_pose_network_placeholders())

        return _prepare_placeholders_dict(
            person_image_in, person_heatmap,
            pose_image_in, pose_centermap_in, pose_heatmap_in)

    def _compute_person_network_placeholders(self):
        size = self._image_properties.get_scaled_size()

        image_in = _prepare_placeholder(1, 3, size)

        heatmap = utils.inference_person(image_in)
        heatmap_large = tf.image.resize_images(heatmap, size)

        return image_in, heatmap_large

    def _compute_pose_network_placeholders(self):
        size = self._image_properties.get_squared_size()

        image_in = _prepare_placeholder(self.POSE_DIM_0, 3, size)
        centermap_in = _prepare_placeholder(self.POSE_DIM_0, 1, size)

        heatmap = utils.inference_pose(image_in, centermap_in)

        return image_in, centermap_in, heatmap


def _prepare_placeholders_dict(
        person_image_in, person_heatmap,
        pose_image_in, pose_centermap_in, pose_heatmap_in):

    placeholders = {
        'person image in': person_image_in,
        'person heatmap': person_heatmap,
        'pose image in': pose_image_in,
        'pose centermap in': pose_centermap_in,
        'pose heatmap': pose_heatmap_in}

    return placeholders


def _prepare_placeholder(dim_0, dim_3, size):
    height = _get_height(size)
    width = _get_width(size)
    placeholder = tf.placeholder(
        tf.float32,
        [dim_0, height, width, dim_3]
    )
    return placeholder
