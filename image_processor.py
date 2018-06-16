import cv2 as cv
import numpy as np
import os


class ImageProcessor(object):
    def __init__(self, input_h, input_w, labels_list, output_path):
        self.input_h = input_h
        self.input_w = input_w
        self.labels_list = labels_list
        self.output_path = output_path
        self.filename = None
        self.img = None
        pass

    def load(self, path, file):
        self.img = cv.imread(os.path.join(path, file))
        resized = cv.resize(self.img, (self.input_h, self.input_w), interpolation=cv.INTER_CUBIC)/255.
        image = np.expand_dims(resized, 0)
        self.filename = file
        shape = self.img.shape[:2]
        return image, shape

    def write_labels(self, predictions, show_images=True, frame_color=(255, 255, 255)):
        predicted_boxes, predicted_likelihoods, predicted_labels = predictions
        try: os.mkdir(self.output_path)
        except: pass
        for i, l_i in reversed(list(enumerate(predicted_labels))):
            label = '{} {:.2f}'.format(self.labels_list[l_i], predicted_likelihoods[i])
            top, left, bottom, right = np.round(predicted_boxes[i]).astype(int)
            print(self.filename, '-', label, '- box [{:d},{:d}] [{:d},{:d}]'.format(left, top, right, bottom))
            cv.rectangle(self.img, (left, top), (right, bottom), frame_color, 3)
            cv.putText(self.img, label, (left, top - 8), cv.FONT_HERSHEY_DUPLEX, 0.7, frame_color, 1, cv.LINE_AA)
        cv.imwrite(os.path.join(self.output_path, self.filename), self.img)
        if show_images:
            cv.imshow('image', self.img)
            cv.waitKey(0)