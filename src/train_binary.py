""" Creates a CNN from model.cnn and trains it"""
from model import cnn  # Import required functions

train_path = "../data/2ary/train/"
test_path = "../data/2ary/test/"
json_file = "../model/model_bin.json"
weights_file = "../model/model_bin.h5"
text_file = "../model/model_bin.txt"
image_file = "../model/model_bin.png"
epochs = 100
batch_size = 16
steps_per_epoch = 1750//batch_size
validation_steps = 30
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape = (224, 224, 3)
target_size = (224, 224)
is_binary = True
classes = 4
optimizer = 'adam'
dropout_rate = 0.25
kwargs = {
    'validation_steps': validation_steps,
    'weights_file': weights_file,
    'json_file': json_file,
    'text_file': text_file,
    'image_file': image_file,
    'test_path': test_path,
    'train_path': train_path,
    'input_shape': input_shape,
    'target_size': target_size,
    'is_binary': is_binary,
    'classes': classes,
    'pool_size': pool_size,
    'kernel_size': kernel_size,
    'epochs': epochs,
    'batch_size': batch_size,
    'steps_per_epoch': steps_per_epoch,
    'optimizer': optimizer,
    'dropout_rate': dropout_rate
}


classifier = cnn.CNN(**kwargs)
classifier.create_model()
classifier.train_model()
classifier.export_model()
