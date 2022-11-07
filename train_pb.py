import os
import argparse
import logging
from datetime import datetime
import pytz
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
tf.debugging.disable_traceback_filtering()

from data import classification_dataset
from model import mobile_net_v2_model, efficient_net_v2_b0_model
from callbacks import save_callback

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')

model_dict = {
    'efficient_net_v2_b0_model': efficient_net_v2_b0_model.prepare,
    'mobile_net_v2_model': mobile_net_v2_model.prepare
}

def train(train_input_dir_path, valid_input_dir_path, classes_txt_path, model_type, epochs, step_size, batch_size,
          test_max_sample_per_classes, image_size, output_dir_path):
    # Download data
    with open(classes_txt_path, 'r') as f:
        classes = f.readlines()
    classes = [label.strip() for label in classes]
    # train
    TrainDataset = type(f'TrainDataset', (classification_dataset.ClassificationDataset,), dict())
    train_dataset = TrainDataset(train_input_dir_path, classes, image_size)
    train_dataset = train_dataset.padded_batch(batch_size=batch_size, padding_values=(
    tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.int32)))
    # valid
    ValidDataset = type(f'ValidDataset', (classification_dataset.TestClassificationDataset,), dict())
    valid_dataset = ValidDataset(valid_input_dir_path, classes, test_max_sample_per_classes, image_size)
    valid_data = classification_dataset.TestClassificationDataset.get_all_data(valid_dataset)

    # prepare model
    model = model_dict[model_type](len(classes), image_size, fine=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=tf.keras.metrics.SparseCategoricalAccuracy())
    # prepare callback
    save_model_dir_path = os.path.join(output_dir_path,
                                       f'{datetime.now(pytz.timezone("Asia/Tokyo")).strftime("%Y-%m-%d-%H-%M-%S")}')
    prefix = f'step-{step_size}_batch-{batch_size}'
    callback = save_callback.SaveCallback(save_model_dir_path=save_model_dir_path, prefix=prefix)

    model.fit_generator(train_dataset, steps_per_epoch=step_size,
                        epochs=epochs,
                        validation_data=valid_data,
                        callbacks=[callback])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train pb')
    parser.add_argument('--train_input_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/train')
    parser.add_argument('--valid_input_dir_path', type=str, default='~/.vaik-mnist-classification-dataset/valid')
    parser.add_argument('--classes_txt_path', type=str, default='~/.vaik-mnist-classification-dataset/classes.txt')
    parser.add_argument('--model_type', type=str, default='efficient_net_v2_b0_model')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--step_size', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--test_max_sample_per_classes', type=int, default=100)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--output_dir_path', type=str, default=os.path.expanduser('~/output_model'))
    args = parser.parse_args()

    args.train_input_dir_path = os.path.expanduser(args.train_input_dir_path)
    args.valid_input_dir_path = os.path.expanduser(args.valid_input_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    os.makedirs(args.output_dir_path, exist_ok=True)
    train(args.train_input_dir_path, args.valid_input_dir_path, args.classes_txt_path, args.model_type,
          args.epochs, args.step_size, args.batch_size, args.test_max_sample_per_classes,
          (args.image_height, args.image_width), args.output_dir_path)
