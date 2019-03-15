import os
from keras import layers, models
from keras.models import Sequential, load_model
from keras.layers import (Conv2D, MaxPooling2D, ReLU, Dropout, Input,
                          Flatten, Dense, LeakyReLU, Softmax,
                          BatchNormalization)
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
from contextlib import redirect_stdout
from keras.applications import InceptionResNetV2
from keras.models import Model
import pandas as pd
import keras_metrics


class CNN:

    def __init__(self, history_file=None,
                 validation_steps=None, weights_file=None, json_file=None,
                 text_file=None, image_file=None, test_path=None,
                 train_path=None, input_shape=(512, 512), dropout_rate=0.3,
                 target_size=(224, 224), is_binary=False, classes=2,
                 pool_size=(2, 2), kernel_size=(3, 3), optimizer='sgd',
                 epochs=30, batch_size=32, steps_per_epoch=130):
        """ Initialise the class with the given Parameters

        Parameters
        ----------
        input_shape: tuple
            It describes the shape of the input image
            Format: (width,height,channels)
            Ex: (512,512)
        target_shape: tuple
            It describes the shape of the image to be fed into the CNN
            Format: (width,height,channels)
            Ex: (512,512)
        is_binary: bool
            Describes if the classifier is binary or categorical
        classes: int ,optional
            The number of classes. Required when mode is set to categorical
        pool_size: tuple
            Size of the pooling kernel
            Format: (width,height)
        kernel_size: tuple
            Size of the convolutional kernel
            Format: (width,height)
        epochs: int
            Number of epochs
        batch_size: int
            Size of each training batch
        steps_per_epoch: int
            Number of steps in each epoch
        validation_steps: int
            Number of steps in each validation epoch
        weights_file: str
            Path to the HDF5 storing the weights
        json_file: str
            Path to the json file storing the model architecture
        text_file: str
            Path to the text file storing the model architecture
        image_file: str
            Path to the image file storing the model architecture
        train_path: str
            Path to the training directory
        test_path: str
            Path to the test directory
        activation: str
            Specifies the activation function
        optimizer: str
            Specifies the optimizer
        dropout_rate: float
            Specifies the Dropout rate

        Returns
        -------
        None

        """
        self.input_shape = input_shape
        self.optimizer = optimizer
        self.dropout_rate = dropout_rate
        if is_binary:
            self.args = {
                'class_mode': 'binary',
                'units': 1,
                'activation': 'sigmoid',
                'loss': 'binary_crossentropy',
                'precision': 'keras_metrics.binary_precision',
                'recall': 'keras_metrics.binary_recall',
                'f1_score': 'keras_metrics.binary_f1_score'
            }
        else:
            self.args = {
                'class_mode': 'categorical',
                'units': classes,
                'activation': 'softmax',
                'loss': 'categorical_crossentropy',
                'precision': 'keras_metrics.categorical_precision',
                'recall': 'keras_metrics.categorical_recall',
                'f1_score': 'keras_metrics.categorical_f1_score'
            }
        self.pool_size = pool_size
        self.kernel_size = kernel_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.steps_per_epoch = steps_per_epoch
        self.validation_steps = validation_steps
        self.weights_file = weights_file
        self.json_file = json_file
        self.text_file = text_file
        self.image_file = image_file
        self.train_path = train_path
        self.test_path = test_path
        self.target_size = target_size
        self.history_file = history_file
        self.history = [0, 0]

    def create_model(self):
        # if os.path.isfile(self.weights_file):
        #     print("Saved model found")
        #     self.model = load_model(self.weights_file)
        #     print("Saved model loaded successfully")
        # else:
        inputs = Input(shape=self.input_shape)
        base_model = InceptionResNetV2(
            include_top=False, weights='imagenet', input_tensor=inputs, input_shape=self.input_shape, pooling='avg', classes=self.args.get('units'))
        # for layer in base_model.layers[:-20]:
        #     layer.trainable = False
        for layer in base_model.layers:
            layer.trainable = True

        t = base_model(inputs)
        # flatten = Flatten(name='Flatten')(t)
        outputs = Dense(self.args.get('units'),
                        activation=self.args.get('activation'))(t)

        self.model = Model(inputs=inputs, outputs=outputs)

        # print(self.model.summary())

    def train_model(self):
        self.model.compile(optimizer=self.optimizer,
                           loss=self.args.get('loss'),
                           metrics=[self.args.get('recall'), self.args.get('precision'), self.args.get('f1_score')])
        checkpoint = ModelCheckpoint(self.weights_file,
                                     monitor="loss",
                                     verbose=1,
                                     save_best_only=True,
                                     mode="min",
                                     save_weights_only=False
                                     )
        callbacks_list = [checkpoint]

        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            shear_range=0.5,
            zoom_range=0.5,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=90,
        )

        train_set = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.args.get('class_mode'),
            subset="training",
        )
        val_set = train_datagen.flow_from_directory(
            self.train_path,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=self.args.get('class_mode'),
            subset="validation",
        )

        self.history_train = self.model.fit_generator(
            train_set, steps_per_epoch=self.steps_per_epoch,
            epochs=self.epochs,
            validation_steps=self.validation_steps,
            validation_data=val_set,
            callbacks=callbacks_list,
        )
        print("Everything's done")

    def evaluate_model(self):
        if os.path.isfile(self.weights_file):
            print("Saved model found")
            self.model = load_model(self.weights_file)
            print("Saved model loaded successfully")
            self.model.compile(optimizer=self.optimizer,
                               loss=self.args.get('loss'),
                               metrics=["accuracy"])
            test_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.5,
                zoom_range=0.5,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=90,
            )
            test_set = test_datagen.flow_from_directory(
                self.test_path,
                target_size=self.target_size,
                batch_size=1,
                class_mode=self.args.get('class_mode'),
            )
            self.history_test = self.model.predict_generator(
                test_set, steps=self.steps_per_epoch)

            print(self.history_test)

        else:
            print("Model not found")

    def export_model(self):
        with open(self.text_file, "w") as my_file:
            with redirect_stdout(my_file):
                self.model.summary()
        plot_model(self.model, to_file=self.image_file,
                   show_layer_names=True, show_shapes=True)
        model_json = self.model.to_json()
        with open(self.json_file, "w") as json_file:
            json_file.write(model_json)
        with open(self.history_file, "w"):
            df = pd.DataFrame(self.history)
            df.to_csv(self.history_file)
