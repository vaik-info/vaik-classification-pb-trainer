import tensorflow as tf
import glob
import os
import re
import random
from tqdm import tqdm
import numpy as np

from data import ops


class ClassificationDataset:
    input_size = None
    classes = None
    image_dict = None
    aug_ratio = 0.5

    def __new__(cls, input_dir_path, classes, input_size=(256, 256)):
        cls.classes = classes
        cls.input_size = input_size
        cls.image_dict = cls._prepare_image_dict(input_dir_path, classes)
        cls.output_signature = (tf.TensorSpec(name=f'image', shape=(input_size[0], input_size[1], 3), dtype=tf.uint8),
                                tf.TensorSpec(name=f'class_index', shape=(), dtype=tf.int32))
        dataset = tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=cls.output_signature
        )
        return dataset

    @classmethod
    def _generator(cls):
        while True:
            class_index = random.choice(list(range(len(cls.classes))))
            class_label = cls.classes[class_index]
            image_path = random.choice(cls.image_dict[class_label])
            tf_image = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
            if random.uniform(0.0, 1.0) < cls.aug_ratio:
                tf_image = cls._data_aug(tf_image.numpy())
            tf_image = tf.image.resize(tf_image, (cls.input_size[0], cls.input_size[1]), preserve_aspect_ratio=True)

            # insert
            canvas_image = tf.image.pad_to_bounding_box(tf_image, 0, 0, cls.input_size[0], cls.input_size[1])
            yield (
                tf.convert_to_tensor(canvas_image),
                tf.convert_to_tensor(class_index)
            )

    @classmethod
    def _prepare_image_dict(cls, input_dir_path, classes):
        image_dict = {}
        for class_label in classes:
            dir_path_list = glob.glob(os.path.join(input_dir_path, f'**/{class_label}/'), recursive=True)
            if len(dir_path_list) > 0:
                dir_path = dir_path_list[0]
            else:
                print(f'not found:{class_label}')
            image_dict[class_label] = []
            image_path_list = [file_path for file_path in glob.glob(os.path.join(dir_path, '**/*.*'), recursive=True) if
                               re.search('.*\.(png|jpg|bmp)$', file_path)]
            for image_path in tqdm(image_path_list, desc=f'_prepare_image_dict:{class_label}'):
                image_dict[class_label].append(image_path)
        return image_dict

    @classmethod
    def _data_aug(cls, np_image: np.array, random_r_ratio=0.25):
        np_image = ops.random_scale(np_image)
        np_image = ops.random_resize(np_image)
        np_image = ops.random_flip(np_image)
        np_image = ops.random_padding(np_image)
        np_image = ops.random_hsv(np_image, random_ratio=random_r_ratio)
        return np_image


class TestClassificationDataset(ClassificationDataset):
    max_sample_per_classes = None

    def __new__(cls, input_dir_path, classes, max_sample_per_classes, input_size=(256, 256)):
        cls.max_sample_per_classes = max_sample_per_classes
        return super(TestClassificationDataset, cls).__new__(cls, input_dir_path, classes, input_size)

    @classmethod
    def _generator(cls):
        for class_index, class_label in enumerate(cls.classes):
            for image_index, image_path in enumerate(cls.image_dict[class_label]):
                if image_index > cls.max_sample_per_classes - 1:
                    break
                tf_image = tf.image.decode_image(tf.io.read_file(image_path), channels=3)
                tf_image = tf.image.resize(tf_image, (cls.input_size[0], cls.input_size[1]), preserve_aspect_ratio=True)
                canvas_image = tf.image.pad_to_bounding_box(tf_image, 0, 0, cls.input_size[0], cls.input_size[1])
                yield (
                    tf.convert_to_tensor(canvas_image),
                    tf.convert_to_tensor(class_index)
                )
    @classmethod
    def get_all_data(cls, dataset):
        dataset = iter(dataset)
        data_list = []
        for data in dataset:
            data_list.append(data)
        all_data_list = [None for _ in range(len(data_list[0]))]
        for index in range(len(data_list[0])):
            all_data_list[index] = tf.stack([data[index] for data in data_list])
        return tuple(all_data_list)

