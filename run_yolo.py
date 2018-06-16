import os
from loader import Loader
from net import Yolo
from image_processor import ImageProcessor

test_path = 'test'
cfg_file = 'yolo.cfg'
weights_file = 'yolo.weights'
prediction_threshold = 0.25
show_images = True

loader = Loader()
yolo = Yolo()

yolo.build(model=loader.load(cfg_file=cfg_file, weights_file=weights_file),
           labels=loader.load_labels(),
           anchors=loader.anchors,
           prediction_threshold=prediction_threshold
           )
im_proc = ImageProcessor(input_h=loader.input_h,
                         input_w=loader.input_w,
                         labels_list=loader.labels,
                         output_path=test_path + '/out'
                         )
print(' ')
print('begin images processing ...')

for image_file in os.listdir(test_path):
    if os.path.isdir(os.path.join(test_path, image_file)):
        continue
    image, shape = im_proc.load(test_path, image_file)
    predictions = yolo.predict(image, shape)
    im_proc.write_labels(predictions=predictions, show_images=show_images)
