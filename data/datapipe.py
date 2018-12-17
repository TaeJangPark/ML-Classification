import glob, os, math, random, csv
import numpy as np
import tensorflow as tf
from collections import defaultdict
from PIL import Image
from configs import config as cfg
from tensorflow.python.ops import control_flow_ops

def _apply_with_random_selector(x, func, num_cases):
    """Computes func(x, sel), with sel sampled from [0...num_cases-1].

    Args:
        x: input Tensor.
        func: Python function to apply.
        num_cases: Python int32, number of cases to sample sel from.

    Returns:
        The result of func(x, sel), where func receives the value of the
        selector as a python integer, but sel is sampled dynamically.
    """
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
                                   for case in range(num_cases)])[0]

def _distort_color(image, color_ordering=0, fast_mode=True, scope=None):
    """
    Args:
        image: 3-D Tensor containing single image in [0, 1].
        color_ordering: Python int, a type of distortion (valid values: 0-3).
        fast_mode: Avoids slower ops (random_hue and random_contrast)
        scope: Optional scope for name_scope.
    Returns:
        3-D Tensor color-distorted image on range [0, 1]
    Raises:
        ValueError: if color_ordering not in [0, 3]
    """
    with tf.name_scope(scope, 'distort_color', [image]):
        if fast_mode:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            else:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
        else:
            if color_ordering == 0:
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
            elif color_ordering == 1:
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
            elif color_ordering == 2:
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
            elif color_ordering == 3:
                image = tf.image.random_hue(image, max_delta=0.2)
                image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
                image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
                image = tf.image.random_brightness(image, max_delta=32. / 255.)
            else:
                raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

def _distorted_bounding_box_crop(image, bbox,
                                 min_object_covered=0.1,
                                 aspect_ratio_range=(0.75, 1.33),
                                 area_range=(0.7, 1.0),
                                 max_attempts=100,
                                 scope=None):
    with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
        # Each bounding box has shape [1, num_boxes, box coords] and
        # the coordinates are ordered [ymin, xmin, ymax, xmax].

        # A large fraction of image datasets contain a human-annotated bounding
        # box delineating the region of the image containing the object of interest.
        # We choose to create a new bounding box for the object which is a randomly
        # distorted version of the human-annotated bounding box that obeys an
        # allowed range of aspect ratios, sizes and overlap with the human-annotated
        # bounding box. If no box is supplied, then we assume the bounding box is
        # the entire image.
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True)
        bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        cropped_image = tf.slice(image, bbox_begin, bbox_size)
        return cropped_image, distort_bbox

def _preprocess_for_training(image, bbox=None):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32, name='original')

    """ step 1. crop the image """
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                           dtype=tf.float32,
                           shape=[1, 1, 4])
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0), bbox)
    tf.summary.image('image_with_bboxes/image_with_bounding_boxes', image_with_box)

    distorted_image, distorted_bbox = _distorted_bounding_box_crop(image, bbox)

    """ step 2. min size resize """

    distorted_image = tf.expand_dims(distorted_image, 0)
    distorted_image = tf.image.resize_bilinear(distorted_image, [cfg.FLAGS.image_min_size, cfg.FLAGS.image_min_size], align_corners=False)
    distorted_image = tf.squeeze(distorted_image, axis=0)
    distorted_image.set_shape([cfg.FLAGS.image_min_size, cfg.FLAGS.image_min_size, 3])
    tf.summary.image('crop_image/cropped_resized_image', tf.expand_dims(distorted_image, 0))

    """ step 3. Randomly flip the image horizontally. """
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    """ step 4. Randomly flip the image vertically. """
    distorted_image = tf.image.random_flip_up_down(distorted_image)

    # """ step 5. Randomly distort the colors. There are 4 ways to do it.  """
    # distorted_image = _apply_with_random_selector(
    #     distorted_image,
    #     lambda x, ordering: _distort_color(x, ordering, fast_mode=True),
    #     num_cases=4)
    #
    tf.summary.image('aug_output/final_distorted_image', tf.expand_dims(distorted_image, 0))

    return distorted_image

def _preprocess_for_test(image):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    """ step 1. resize image """
    image = tf.expand_dims(image, 0)
    image = tf.image.resize_bilinear(image, [cfg.FLAGS.image_min_size, cfg.FLAGS.image_min_size], align_corners=False)
    image = tf.squeeze(image, axis=[0])

    return image


import pandas as pd
def get_dataset(split_name, dataset_dir, im_batch=4):
    name2cls = {}
    fp = open('data/train_class_names.csv')
    train_class_info = csv.reader(fp)
    """ Image label, class_ID, class_Name """
    for i, line in enumerate(train_class_info):
        # print(line)
        cls_label = line[0]
        cls_name = line[2]
        name2cls[cls_name] = cls_label
    fp.close()

    filenames = []
    gt_labels = []
    with open('data/train_image_labels.csv', 'r', newline='') as f:
        rf = csv.reader(f)
        for i, line in enumerate(rf):
            im_path = os.path.join(dataset_dir, split_name, line[1]+'.jpg')
            attrs = line[2:]
            labels = []
            for att in attrs:
                cls_id = int(name2cls[att])
                labels.append(cls_id)

            filenames.append(im_path)
            gt_labels.append(labels)
    f.close()

    print(len(gt_labels), len(filenames), filenames[0], gt_labels[0])

    # images = tf.convert_to_tensor(filenames)
    # labels = tf.convert_to_tensor(gt_labels)
    # input_queue = tf.train.slice_input_producer([images, labels])
    # image = tf.read_file(input_queue[0])
    # image = tf.image.decode_jpeg(image, channels=3)
    # # label = tf.string_to_number(input_queue[1], out_type=tf.int32)
    # label = input_queue[1]
    #
    # if split_name is 'train':
    #     image = _preprocess_for_training(image)
    # if split_name is 'test':
    #     image = _preprocess_for_test(image)
    #
    # min_after_dequeue = 3 * im_batch
    # capacity = min_after_dequeue * 2
    # batches = tf.train.shuffle_batch([image, label],
    #                                  batch_size=im_batch,
    #                                  capacity=capacity,
    #                                  min_after_dequeue=min_after_dequeue,
    #                                  num_threads=cfg.FLAGS.num_preprocessing_threads)
    #
    # tf.summary.image(name='distorted_input', tensor=batches[0], max_outputs=1)

    # return batches[0], batches[1]
    return filenames, gt_labels

def test(split_name, dataset_dir):
    image_dir = os.path.join(dataset_dir, split_name)
    image_list = [f.split('.')[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
    image_label_list = list(set(image_list))
    print('total number of the train_images: ', len(image_list))

    class_name2id = {}
    class_id2name = {}
    classes_dict = open(os.path.join(dataset_dir, 'class-descriptions.csv'))
    class_desc = csv.reader(classes_dict)
    for i, line in enumerate(class_desc):
        class_name2id[line[0]] = i
        class_id2name[i] = line[1]

    class_idx = [0 for _ in range(len(class_name2id.keys()))]
    print('total number of the train classes : ', len(class_idx))

    f = open('data/train_classes.txt', 'w')
    image_gt_labels = defaultdict(list)
    train_list = open(os.path.join(dataset_dir, 'train-annotations-human-imagelabels.csv'))
    train_f = csv.reader(train_list)

    for i, line in enumerate(train_f):
        if i == 0:
            print(line)
            continue
        image_name = line[0]
        if int(line[3]) > 0:
            label_id = class_name2id[line[2]]
            image_gt_labels[image_name].append(label_id)
            if class_idx[label_id] == 0:
                class_idx[label_id] = 1
            f.write(line[2]+' '+str(label_id))
            f.write('\n')
    f.close()
    cnt = 0
    with open('data/train_class_names.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        for k, v in class_name2id.items():
            if v<10:
                print(k, class_id2name[v])
            if class_idx[v] > 0:
                ss = [cnt] + [k] + [class_id2name[v]]
                wr.writerow(ss)
                cnt+=1

    f.close()
    print(cnt)
    sss

    with open('data/train_image_labels.csv', 'w', newline='') as f:
        wr = csv.writer(f)
        index = 0
        for n in image_label_list:
            ss = [index, n]
            for id in image_gt_labels[n]:
                ss += [class_id2name[id]]

            wr.writerow(ss)
            index += 1
    f.close()
    print(index)
    sss


