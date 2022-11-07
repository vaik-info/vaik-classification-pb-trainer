import os
import tensorflow as tf
from sklearn import metrics
import numpy as np
import copy

class SaveCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_model_dir_path, prefix):
        super(SaveCallback, self).__init__()
        os.makedirs(save_model_dir_path, exist_ok=True)

        self.save_model_dir_path = save_model_dir_path
        self.prefix = prefix

    def on_epoch_end(self, epoch, logs=None):
        # create save model dir
        loss_string = "_".join([f'{k}_{v:.4f}' for k, v in logs.items()])
        save_model_name = f'{self.prefix}_epoch-{epoch}_{loss_string}'
        output_model_dir_path = os.path.join(self.save_model_dir_path, save_model_name)
        os.makedirs(output_model_dir_path, exist_ok=True)

        # save model
        self.model.save(os.path.join(f'{output_model_dir_path}', 'model'))