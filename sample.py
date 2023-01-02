

# https://stackoverflow.com/questions/74725957/indexerror-tuple-index-out-of-range-for-tensorflow-machine-learning-code
# https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images?resource=download

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import matplotlib.pyplot as plt

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Function
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
def custom_image_preprocess( image ):

	random_lotation_layer = tf.keras.layers.RandomRotation(
							factor=(-0.2, 0.3),
							fill_mode='nearest',
							interpolation='nearest',
							seed=None,
							fill_value=0.0,
						)
	
	return	random_lotation_layer(image)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
BATCH_SIZE = 16
IMG_HEIGHT = 72
IMG_WIDTH = 72
IMG_CHANNELS= 3

train_dir = "F:\\datasets\\Alzheimer's Dataset\\train"
test_dir = "F:\\datasets\\Alzheimer's Dataset\\test"

seed_1 = tf.random.set_seed(1234)
seed_2 = tf.random.set_seed(1235)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: DataSet
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
train_image_generator = ImageDataGenerator(rescale=1. / 255, vertical_flip=True, horizontal_flip=True, preprocessing_function=custom_image_preprocess,) 
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
	class_mode='binary',
    color_mode='rgb',
	seed=seed_1,)
	
test_image_generator = ImageDataGenerator(rescale=1. / 255, vertical_flip=True, horizontal_flip=True, preprocessing_function=custom_image_preprocess,)
test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=test_dir,
    shuffle=False,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
	class_mode='binary',
    color_mode='rgb',
	seed=seed_2,)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Initialize
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
base_model = tf.keras.applications.Xception( weights='imagenet', input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), include_top=False)  
base_model.trainable = False
inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

x = tf.keras.applications.xception.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Complie
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=tf.keras.metrics.BinaryAccuracy())

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Model Training
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
model.fit(train_data_gen, epochs=10, validation_data=test_data_gen)
