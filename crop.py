import os
import pandas as pd
from ast import literal_eval as make_tuple
from functools import partial


import tensorflow as tf
def random_zoom(image, zoom_range=0.25):
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    scale = tf.random.uniform([], 1.0 - zoom_range, 1.0)
    new_h = tf.cast(scale * tf.cast(h, tf.float32), tf.int32)
    new_w = tf.cast(scale * tf.cast(w, tf.float32), tf.int32)

    image = tf.image.resize_with_crop_or_pad(image, new_h, new_w)
    image = tf.image.resize(image, [h, w])
    return image

def augment_image(image, label, shear_range=0.3, zoom_range=0.25):
    # Normalize image to float32
    image = tf.image.convert_image_dtype(image, tf.float32)


    # Random zoom
    image = random_zoom(image, zoom_range)

    # Random horizontal flip
    image = tf.image.random_flip_left_right(image)
    #image = tf.image.resize(image, [224, 224])

    return image, label


def pad_bbox(bbox, pad):
    # cvzone bbox - [x, y, w, h]
    pad_w = int(bbox[2] * pad)
    pad_h = int(bbox[3] * pad)
    return [max(bbox[0]-pad_w, 0),
            max(bbox[1]-pad_h, 0),
            bbox[2]+pad_w,
            bbox[3]+pad_h
            ]



def load_and_crop_image(image_path, bbox, label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)

    # Convert bbox to int if it's not normalized
    ymin, xmin, ymax, xmax = tf.unstack(bbox)
    ymin = tf.cast(ymin, tf.int32)
    xmin = tf.cast(xmin, tf.int32)
    h = tf.cast(ymax, tf.int32)
    w = tf.cast(xmax, tf.int32)
    # Crop
    cropped = image[xmin: xmin+w, ymin:ymin+h,:]
    resized = tf.image.resize(cropped, [224, 224])
    resized /= 255

    return resized, label

def get_dataset(df, root, pad=0.2):
    ## bboxes in format [x, y, w, h]
    df['paths'] = df['paths'].apply(lambda x: os.path.join(root, x))
    image_paths = df['paths'].to_list()
    bboxes = df['bbox'].apply(lambda x: make_tuple(x)[0]['bbox']).to_list()
    padded_fn = partial(pad_bbox, pad=pad)
    bboxes = list(map(padded_fn, bboxes))
    labels = df['label'].to_list()
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, bboxes, labels))

    dataset = dataset.map(lambda path, box, label: load_and_crop_image(path, box, label))
    dataset = dataset.map(lambda img, label: augment_image(img, label))
    dataset = dataset.apply(tf.data.experimental.ignore_errors())

    dataset = dataset.shuffle(True).batch(32)

    return dataset