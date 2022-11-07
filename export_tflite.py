import glob
import os
import argparse
import random

import tensorflow as tf


def tf2tflite(input_model_dir_path, output_model_file_path, representative_dataset_gen):
    os.makedirs(os.path.dirname(output_model_file_path), exist_ok=True)
    tf.compat.v1.enable_eager_execution()

    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_dir_path)
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.target_spec.supported_types = [tf.int8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8
    converter.representative_dataset = representative_dataset_gen

    tflite_quant_model = converter.convert()
    open(output_model_file_path, "wb").write(tflite_quant_model)


def main(input_model_dir_path, train_input_dir_path, output_model_file_path, sample_max_num):
    def representative_dataset_gen():
        for step_index, image_path in enumerate(image_path_list):
            if step_index > sample_max_num:
                break
            yield [
                tf.cast(tf.expand_dims(tf.image.decode_image(tf.io.read_file(image_path), channels=3), 0), tf.float32)]

    types = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
    image_path_list = []
    for files in types:
        image_path_list.extend(glob.glob(os.path.join(train_input_dir_path, files), recursive=True))
    random.shuffle(image_path_list)

    tf2tflite(input_model_dir_path, output_model_file_path, representative_dataset_gen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='export')
    parser.add_argument('--input_model_dir_path', type=str,
                        default='~/output_model/2022-11-07-15-55-32/step-1000_batch-8_epoch-9_loss_0.1008_sparse_categorical_accuracy_0.9715_val_loss_1.3999_val_sparse_categorical_accuracy_0.6440',
                        help="input tensor model dir path")
    parser.add_argument('--train_input_dir_path', type=str,
                        default=os.path.expanduser('~/.vaik-mnist-classification-dataset/dump'))
    parser.add_argument('--output_model_file_path', type=str,
                        default='~/output_tflite_model/mnist_mobile_net_v2.tflite',
                        help="output tflite model dir path")
    parser.add_argument('--sample_max_num', type=int, default=25000, help="output tflite model dir path")
    args = parser.parse_args()

    args.input_model_dir_path = os.path.expanduser(args.input_model_dir_path)
    args.train_input_dir_path = os.path.expanduser(args.train_input_dir_path)
    args.output_model_file_path = os.path.expanduser(args.output_model_file_path)

    main(args.input_model_dir_path, args.train_input_dir_path, args.output_model_file_path)
