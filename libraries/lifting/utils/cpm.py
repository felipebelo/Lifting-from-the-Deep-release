import tensorflow as tf
import tensorflow.contrib.layers as layers

__all__ = [
    'inference_person',
    'inference_pose'
]

INFERENCE_PERSON_LAYERS_DESCRIPTION = [
    {
        'type': 'conv',
        'key': 'conv1_1',
        'vals': [64, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv1_2',
        'vals': [64, 3, 1],
        'rectified': True
    },
    {
        'type': 'pool_stage',
        'key': 'pool1_stage1',
        'vals': [2, 2]
    },

    {
        'type': 'conv',
        'key': 'conv2_1',
        'vals': [128, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv2_2',
        'vals': [128, 3, 1],
        'rectified': True
    },
    {
        'type': 'pool_stage',
        'key': 'pool2_stage1',
        'vals': [2, 2]
    },
    {
        'type': 'conv',
        'key': 'conv3_1',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv3_2',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv3_3',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv3_4',
        'vals': [256, 3, 1],
        'rectified': True
    },

    {
        'type': 'pool_stage',
        'key': 'pool3_stage1',
        'vals': [2, 2]
    },
    {
        'type': 'conv',
        'key': 'conv4_1',
        'vals': [512, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_2',
        'vals': [512, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_3',
        'vals': [512, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_4',
        'vals': [512, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv5_1',
        'vals': [512, 3, 1],
        'rectified': True
    },

    {
        'type': 'conv',
        'key': 'conv5_2_CPM',
        'vals': [128, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv6_1_CPM',
        'vals': [512, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv6_2_CPM',
        'vals': [1, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage2',
        'layers': ['conv6_2_CPM', 'conv5_2_CPM'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage2',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage2',
        'vals': [1, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage3',
        'layers': ['Mconv7_stage2', 'conv5_2_CPM'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage3',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage3',
        'vals': [1, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage4',
        'layers': ['Mconv7_stage3', 'conv5_2_CPM'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage4',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage4',
        'vals': [1, 1, 1],
        'rectified': False
    }
]

INFERENCE_POSE_LAYERS_DESCRIPTION = [
    {
        'type': 'conv',
        'key': 'conv1_1',
        'vals': [64, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv1_2',
        'vals': [64, 3, 1],
        'rectified': True
    },
    {
        'type': 'pool_stage',
        'key': 'pool1_stage1',
        'vals': [2, 2]
    },
    {
        'type': 'conv',
        'key': 'conv2_1',
        'vals': [128, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv2_2',
        'vals': [128, 3, 1],
        'rectified': True
    },
    {
        'type': 'pool_stage',
        'key': 'pool2_stage1',
        'vals': [2, 2]
    },
    {
        'type': 'conv',
        'key': 'conv3_1',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv3_2',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv3_3',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv3_4',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'pool_stage',
        'key': 'pool3_stage1',
        'vals': [2, 2]
    },
    {
        'type': 'conv',
        'key': 'conv4_1',
        'vals': [512, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_2',
        'vals': [512, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_3_CPM',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_4_CPM',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_5_CPM',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_6_CPM',
        'vals': [256, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv4_7_CPM',
        'vals': [128, 3, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv5_1_CPM',
        'vals': [512, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'conv5_2_CPM',
        'vals': [15, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage2',
        'layers': ['conv5_2_CPM', 'conv4_7_CPM', 'pool_center_lower'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage2',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage2',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage2',
        'vals': [15, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage3',
        'layers': ['Mconv7_stage2', 'conv4_7_CPM', 'pool_center_lower'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage3',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage3',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage3',
        'vals': [15, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage4',
        'layers': ['Mconv7_stage3', 'conv4_7_CPM', 'pool_center_lower'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage4',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage4',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage4',
        'vals': [15, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage5',
        'layers': ['Mconv7_stage4', 'conv4_7_CPM', 'pool_center_lower'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage5',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage5',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage5',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage5',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage5',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage5',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage5',
        'vals': [15, 1, 1],
        'rectified': False
    },
    {
        'type': 'concat',
        'key': 'concat_stage6',
        'layers': ['Mconv7_stage5', 'conv4_7_CPM', 'pool_center_lower'],
        'vals': [3]
    },
    {
        'type': 'conv',
        'key': 'Mconv1_stage6',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv2_stage6',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv3_stage6',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv4_stage6',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv5_stage6',
        'vals': [128, 7, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv6_stage6',
        'vals': [128, 1, 1],
        'rectified': True
    },
    {
        'type': 'conv',
        'key': 'Mconv7_stage6',
        'vals': [15, 1, 1],
        'rectified': False
    }
]


def inference_person(image):
    layers_description = INFERENCE_PERSON_LAYERS_DESCRIPTION

    with tf.variable_scope('PersonNet'):
        layers_map = (
            _prepare_network(image, layers_description, {}))

    return _get_last_layer(layers_description, layers_map)


def inference_pose(image, center_map):
    layers_description = INFERENCE_POSE_LAYERS_DESCRIPTION

    with tf.variable_scope('PoseNet'):

        initial_layers_map = {
            'pool_center_lower': layers.avg_pool2d(
                center_map, 9, 8, padding='SAME')}

        layers_map = (
            _prepare_network(
                image, layers_description,
                initial_layers_map))

    return _get_last_layer(layers_description, layers_map)


def _prepare_network(inputs, layers_description, layers_map):
    previous_layer = inputs
    for description in layers_description:
        _prepare_layer(layers_map, description, previous_layer)
        previous_layer = layers_map[description['key']]

    return layers_map


def _get_last_layer(layers_description, layers_map):
    last_layer_description = layers_description[-1]
    last_layer_key = last_layer_description['key']
    last_layer = layers_map[last_layer_key]
    return last_layer


def _prepare_layer(layers_map, description, previous_layer):
    layer_type = description['type']

    if layer_type == 'conv':
        prepare_conv_layer(description, layers_map, previous_layer)

    if layer_type == 'pool_stage':
        prepare_pool_state_layer(description, layers_map, previous_layer)

    if layer_type == 'concat':
        prepare_concat_layer(description, layers_map)


def prepare_concat_layer(description, layers_map):
    scope = description['key']
    vals = description['vals']
    layer_keys = description['layers']
    layers_list = [layers_map[key] for key in layer_keys]
    layers_map[scope] = tf_concat(layers_list, vals)


def prepare_pool_state_layer(description, layers_map, previous_layer):
    scope = description['key']
    vals = description['vals']
    layers_map[scope] = tf_pool_stage(previous_layer, vals)


def prepare_conv_layer(description, layers_map, previous_layer):
    scope = description['key']
    vals = description['vals']
    rectified = description['rectified']
    layers_map[scope] = tf_conv_2d(
        previous_layer, scope, vals, rectified=rectified)


def tf_concat(layers_list, vals):
    concat = tf.concat(layers_list, vals[0])
    return concat


def tf_pool_stage(conv1_2, vals):
    pool_stage = layers.max_pool2d(conv1_2, vals[0], vals[1])
    return pool_stage


def tf_conv_2d(previous_layer, scope, vals, rectified=True):
    conv = layers.conv2d(
        previous_layer,
        vals[0], vals[1], vals[2],
        activation_fn=None, scope=scope)
    if rectified:
        conv = tf.nn.relu(conv)
    return conv
