import tensorflow as tf
from keras import backend as K


class Yolo(object):
    def __init__(self, ):
        self.model = None
        self.labels_list = None
        self.prediction_threshold = None
        self.anchors = None
        self.out = None
        self.input_image_shape = K.placeholder(shape=(2,))
        self.sess = K.get_session()

    def build(self, model, labels, anchors, prediction_threshold=0.25):
        self.model = model
        self.labels_list = labels
        self.prediction_threshold = prediction_threshold
        self.anchors = anchors
        # yolo branch (computational graph)
        conv_h = K.shape(self.model.output)[1:3][0]
        conv_w = K.shape(self.model.output)[1:3][1]
        h_1D_tensor = K.arange(0, stop=conv_h)
        h_1D_tensor = K.tile(h_1D_tensor, [conv_w])
        w_1D_tensor = K.arange(0, stop=conv_w)
        w_1D_tensor = K.tile(K.expand_dims(w_1D_tensor, 0), [conv_h, 1])
        w_1D_tensor = K.flatten(K.transpose(w_1D_tensor))
        conv_tensor = K.transpose(K.stack([h_1D_tensor, w_1D_tensor]))
        conv_tensor = K.reshape(conv_tensor, [1, conv_h, conv_w, 1, 2])
        conv_tensor = K.cast(conv_tensor, K.dtype(self.model.output))
        net_output = K.reshape(self.model.output, [-1, conv_h, conv_w, 5, len(self.labels_list) + 5])
        boxes_location = K.sigmoid(net_output[..., :2])
        boxes_size = K.exp(net_output[..., 2:4])
        boxes_confidence = K.sigmoid(net_output[..., 4:5])
        boxes_likelihoods = K.softmax(net_output[..., 5:])
        # rearrange with anchors
        net_output_rearranged = K.cast(K.reshape(K.shape(net_output)[1:3], [1, 1, 1, 1, 2]), K.dtype(net_output))
        boxes_location = (boxes_location + conv_tensor) / net_output_rearranged
        boxes_size = boxes_size * K.reshape(K.variable(self.anchors), [1, 1, 1, 5, 2]) / net_output_rearranged
        # get raw predictions
        lefts = boxes_location - (boxes_size / 2.)
        rights = boxes_location + (boxes_size / 2.)
        frames = K.concatenate([lefts[..., 1:2], lefts[..., 0:1], rights[..., 1:2], rights[..., 0:1]])
        boxes_scores = boxes_confidence * boxes_likelihoods
        raw_labels = K.argmax(boxes_scores, axis=-1)
        max_box_scores = K.max(boxes_scores, axis=-1)
        prediction_mask = max_box_scores >= self.prediction_threshold
        raw_boxes = tf.boolean_mask(frames, prediction_mask)
        raw_likelihoods = tf.boolean_mask(max_box_scores, prediction_mask)
        raw_labels = tf.boolean_mask(raw_labels, prediction_mask)
        # resizing back
        img_previous_shape = K.stack([self.input_image_shape[0], self.input_image_shape[1],
                              self.input_image_shape[0], self.input_image_shape[1]])
        img_previous_shape = K.reshape(img_previous_shape, [1, 4])
        raw_boxes *= img_previous_shape
        # non-max suppression
        sess2 = K.get_session()
        nms_placeholder = K.variable(10, dtype='int32')
        sess2.run(tf.variables_initializer([nms_placeholder]))
        nms_list = tf.image.non_max_suppression(raw_boxes, raw_likelihoods, nms_placeholder, iou_threshold=0.5)
        likelihoods = K.gather(raw_likelihoods, nms_list)
        boxes = K.gather(raw_boxes, nms_list)
        labels = K.gather(raw_labels, nms_list)
        self.out = [boxes, likelihoods, labels]

    def predict(self, image, shape):
        out_boxes, out_likelihoods, out_labels = self.sess.run(
            self.out,
            feed_dict={
                self.model.input: image,
                self.input_image_shape: shape,
                K.learning_phase(): 0
            })
        return out_boxes, out_likelihoods, out_labels
