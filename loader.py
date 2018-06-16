import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
import configparser
import io


class Loader(object):
    def __init__(self):
        self.weights_file = None
        self.cfg_file = None
        self.anchors = None
        self.input_h = 0
        self.input_w = 0
        self.labels = []

    def load(self, cfg_file='yolo.cfg', weights_file='yolo.weights'):
        last_layer = None
        net_whole = None
        self.weights_file = weights_file
        self.cfg_file = cfg_file
        print('\nloading', self.weights_file, '...')
        weights_data = open(self.weights_file, 'rb')
        weights_data.read(16)
        cfg = configparser.ConfigParser()
        cfg.read_file(self.configToDict())
        for section in cfg.sections():
            print(section, end=' ')
            # section
            if section.startswith('route'):
                idx = [int(i) for i in cfg[section]['layers'].split(',')]
                layers = [net_whole[i] for i in idx]
                if len(layers) > 1:
                    concatenate_layer = concatenate(layers)
                    last_layer = concatenate_layer
                else:
                    last_layer = layers[0]
                net_whole.append(last_layer)
                print(' ')
            # section
            elif section.startswith('reorg'):
                net_whole.append(Lambda(self.space_to_depth_x2,
                                        output_shape=self.space_to_depth_x2_output_shape,
                                        name='space_to_depth_x2')(last_layer))
                last_layer = net_whole[-1]
                print(' ')
            # section
            elif section.startswith('region'):
                anchors_str = cfg[section]['anchors']
                self.anchors = np.array(anchors_str.split(','), dtype=np.float16).reshape(-1, 2)
                print('anchors parsing')
            # section
            elif section.startswith('net'):
                self.input_h = int(cfg[section]['height'])
                self.input_w = int(cfg[section]['width'])
                last_layer = Input(shape=(self.input_h, self.input_w, 3))
                net_whole = [last_layer]
                print((self.input_w,self.input_h))
            # section
            elif section.startswith('convol'):
                activation = cfg[section]['activation']
                print(activation, end='')
                filters = int(cfg[section]['filters'])
                stride = int(cfg[section]['stride'])
                size = int(cfg[section]['size'])
                batch_normalize = 'batch_normalize' in cfg[section]
                weights_shape = (size, size, K.int_shape(last_layer)[-1], filters)
                bias = np.ndarray(
                    shape=(filters,),
                    dtype='float32',
                    buffer=weights_data.read(filters * 4))
                if batch_normalize:
                    print(' BN ', end='')
                    bn_weights = np.ndarray(
                        shape=(3, filters),
                        dtype='float32',
                        buffer=weights_data.read(filters * 12))
                print(weights_shape)
                weights_conv = np.ndarray(
                    shape=(filters, weights_shape[2], size, size),
                    dtype='float32',
                    buffer=weights_data.read(4*np.product(weights_shape)))
                weights_conv = np.transpose(weights_conv, [2, 3, 1, 0])
                if batch_normalize:
                    weights_conv = [weights_conv]
                else:
                    weights_conv = [weights_conv, bias]
                conv_layer = (Conv2D(filters, (size, size),
                                     strides=(stride, stride),
                                     use_bias=not batch_normalize,
                                     weights=weights_conv,
                                     activation=None,
                                     padding='same'))(last_layer)
                if batch_normalize:
                    conv_layer = (BatchNormalization(weights=[bn_weights[0],  # gamma
                                                              bias,  # beta
                                                              bn_weights[1],  # mu
                                                              bn_weights[2]  # sigma
                                                             ]))(conv_layer)
                last_layer = conv_layer
                if activation == 'leaky':
                    last_layer = LeakyReLU(alpha=0.1)(last_layer)
                net_whole.append(last_layer)
            # section
            elif section.startswith('maxpool'):
                size = int(cfg[section]['size'])
                stride = int(cfg[section]['stride'])
                net_whole.append(MaxPooling2D(padding='same',
                                               pool_size=(size, size),
                                               strides=(stride, stride))(last_layer))
                last_layer = net_whole[-1]
                print(' ')
        weights_data.close()
        return Model(inputs=net_whole[0], outputs=net_whole[-1])

    def load_labels(self, path_labels='labels.txt'):
        f = open(path_labels)
        self.labels = [str.strip() for str in f]
        f.close()
        return self.labels

    def configToDict(self):
        counter = 0
        stream = io.StringIO()
        f = open(self.cfg_file)
        for string in f:
            if string.startswith('['):
                old_str = string.strip().strip('[]')
                new_str = old_str + '_' + str(counter)
                counter += 1
                string = string.replace(old_str, new_str)
            stream.write(string)
        stream.seek(0)
        f.close()
        return stream

    def space_to_depth_x2(self, x):
        return tf.space_to_depth(x, block_size=2)

    def space_to_depth_x2_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] // 2, input_shape[2] // 2,
                4 * input_shape[3]) if input_shape[1] else (input_shape[0], None, None, 4 * input_shape[3])
