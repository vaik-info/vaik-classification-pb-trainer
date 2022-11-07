import os
import argparse
from tqdm import tqdm
from PIL import Image
import tensorflow as tf

from data import classification_dataset


def dump(input_image_dir_path, classes_txt_path, sample_num, image_height, image_width, output_dir_path):
    os.makedirs(output_dir_path, exist_ok=True)
    classes = []
    with open(classes_txt_path) as f:
        for line in f:
            classes.append(line.strip())

    TrainDataset = type(f'TrainDataset', (classification_dataset.ClassificationDataset,), dict())
    train_dataset = TrainDataset(input_image_dir_path, classes, (image_height, image_width))
    train_dataset = train_dataset.padded_batch(batch_size=1, padding_values=(tf.constant(0, dtype=tf.uint8), tf.constant(0, dtype=tf.int32)))
    train_dataset = iter(train_dataset)
    for index in tqdm(range(sample_num)):
        image, class_index = train_dataset.get_next()
        class_index = class_index.numpy()[0]
        image = image.numpy()[0]
        output_image_path = os.path.join(output_dir_path, f'{classes[class_index]}_{index:09d}.jpg')
        Image.fromarray(image).save(output_image_path, quality=100, subsampling=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--input_image_dir_path', type=str,
                        default=os.path.expanduser('~/.vaik-mnist-classification-dataset/train'))
    parser.add_argument('--classes_txt_path', type=str,
                        default=os.path.expanduser('~/.vaik-mnist-classification-dataset/classes.txt'))
    parser.add_argument('--sample_num', type=int, default=25000)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--output_dir_path', type=str,
                        default=os.path.expanduser('~/.vaik-mnist-classification-dataset/dump'))
    args = parser.parse_args()

    args.input_image_dir_path = os.path.expanduser(args.input_image_dir_path)
    args.classes_txt_path = os.path.expanduser(args.classes_txt_path)
    args.output_dir_path = os.path.expanduser(args.output_dir_path)

    dump(args.input_image_dir_path, args.classes_txt_path, args.sample_num, args.image_height, args.image_width, args.output_dir_path)
