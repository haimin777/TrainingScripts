import os
import sys
import pandas as pd
import json

import tensorflow as tf
import pandas as pd
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import math
import datetime
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
from crop import get_dataset as crop_dataset


def prepare_data(dataset_dir, trn_df, tst_df, batch_size=32):


    datagen_trn = ImageDataGenerator(
        #shear_range=0.3,
        zoom_range=0.15,

        horizontal_flip=True,
        preprocessing_function=preprocess_input
    )

    datagen_tst = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )
    train_gen = datagen_trn.flow_from_dataframe(
        dataframe=trn_df,
        directory=dataset_dir,
        x_col='paths',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='raw',  # 'categorical' if multi-class one-hot
        shuffle=True
    )
    test_gen = datagen_tst.flow_from_dataframe(
        dataframe=tst_df,
        directory=dataset_dir,
        x_col='paths',
        y_col='label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='raw',  # 'categorical' if multi-class one-hot
        shuffle=True
    )

    return train_gen, test_gen

def prepare_model(train_samples_num, num_epoch, batch_size, img_size, model=0):
    '''
    model: 0 EfnrtB0
    model: 1 EfnetB1
    model: 2 EfnetB2
    '''

    model_type = {0: EfficientNetV2B0,
                  1: EfficientNetV2B1,
                  2: EfficientNetV2B2}
    # Load EfficientNetB0 without top (no classifier head)
    #base_model = EfficientNetV2B0(
    base_model = model_type[model](    
        include_top=False,
        weights='imagenet',
        input_shape=(img_size, img_size, 3),
        pooling='avg'  # GlobalAveragePooling2D
    )

    # Freeze base model (optional)
    # base_model.trainable = False

    # Add custom head
    x = base_model.output
    x = layers.Dropout(0.3)(x)
    output = layers.Dense(1, activation='sigmoid')(x)  # 1 output for binary classification



    # Create final model
    model = models.Model(inputs=base_model.input, outputs=output)

    initial_learning_rate = 0.001
    
    #decay_steps = (num_samples // batch_size) * num_epochs

    decay_steps = (train_samples_num // batch_size) * num_epoch  # or total_steps = epochs * steps_per_epoch

    alpha = 1e-9  # Minimum learning rate as a fraction of initial (0 means decay to 0)

    lr_schedule = CosineDecay(
        initial_learning_rate=initial_learning_rate,
        decay_steps=decay_steps,
        alpha=alpha
    ) 
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)


    # Compile (example)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    return model


class ImageLogger(tf.keras.callbacks.Callback):
    def __init__(self, log_dir, data_generator, num_images=3):
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir + '/images')
        self.data_gen = data_generator
        self.num_images = num_images

    def on_epoch_end(self, epoch, logs=None):
        # Only after first epoch
        if epoch == 0:
            # Get a batch
            x_batch, y_batch = next(iter(self.data_gen))

            # Run prediction
            preds = self.model.predict(x_batch[:self.num_images])

            # Convert to images for TensorBoard
            with self.file_writer.as_default():
                for i in range(self.num_images):
                    fig, ax = plt.subplots(1, 2, figsize=(6,3))
                    ax[0].imshow(x_batch[i].astype("uint8"))
                    ax[0].set_title("Input")
                    ax[0].axis("off")
                    
                    ax[1].imshow(preds[i].squeeze(), cmap="gray")
                    ax[1].set_title("Prediction")
                    ax[1].axis("off")

                    # Convert figure to image
                    buf = io.BytesIO()
                    plt.savefig(buf, format='png')
                    plt.close(fig)
                    buf.seek(0)
                    image = tf.image.decode_png(buf.getvalue(), channels=4)
                    image = tf.expand_dims(image, 0)  # add batch dimension
                    
                    # Write to TensorBoard
                    tf.summary.image(f"Sample_{i}", image, step=epoch)


def main(dataset_dir, config_path):

    with open(config_path) as f:

        config = json.load(f)

    trn_path = config["trn_csv"]
    tst_path = config["tst_csv"]
    num_epoch = config['epochs']
    batch_size = config['batch_size']
    img_size = config['img_size']
    model_type = config['model']
    train_on_crop = config['crop']
    pad = config['pad']
    debug = config['debug']
    trn_df = pd.read_csv(trn_path)
    tst_df = pd.read_csv(tst_path)
    if debug:
        trn_df = trn_df[:500]
        tst_df = tst_df[:100]

    if train_on_crop:
        trn_gen = crop_dataset(trn_df, dataset_dir, pad)
        tst_gen = crop_dataset(tst_df, dataset_dir, pad)
    else:
        trn_gen, tst_gen = prepare_data(dataset_dir, trn_df, tst_df, batch_size)

    model = prepare_model(trn_df.shape[0],
                           num_epoch,
                             batch_size,
                             img_size,
                             model_type
                             )

    # Create folders
    checkpoint_dir = "checkpoints/exper_"+os.path.basename(config_path)
    log_dir = f"logs/exper_{os.path.basename(config_path)}/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Define checkpoint path (only saves weights)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "model_epoch_{epoch:02d}_valacc_{val_accuracy:.2f}.weights.h5"),
        monitor='val_accuracy',
        save_best_only=False,
        save_weights_only=True,
        verbose=1
    )
    class LRSchedulerLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            # Handle both constant and scheduled learning rates
            lr_schedule = self.model.optimizer.learning_rate
            if callable(lr_schedule):
                step = self.model.optimizer.iterations.numpy()
                lr = lr_schedule(step).numpy()
            else:
                lr = tf.keras.backend.get_value(lr_schedule)
            print(f"\n📘 Epoch {epoch + 1} — Learning Rate: {lr:.6f}")


    # TensorBoard callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    image_logger = ImageLogger(log_dir, tst_gen, num_images=3)

    history = model.fit(
                trn_gen,
                validation_data=tst_gen,
                epochs=num_epoch,
                callbacks=[
                    checkpoint_callback,
                    tensorboard_callback,
                    LRSchedulerLogger(),
                    image_logger
                ]
)


if __name__ == "__main__":

    main(sys.argv[1], sys.argv[2])    