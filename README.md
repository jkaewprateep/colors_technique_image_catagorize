# colors_technique_image_catagorize
For the study of color technique, image categorize, and binary image categorizes, in traditional medical treatments fundamentals and services are best efforts by medical resources and processes. The instruments and tools working on the accuracy results, the binary cross entropy is reserved for the task that is not because it looks old but best efforts with tools model builders and research awareness of this to selected Binary CrossEntrophy.

## DataSet ##

The Image Generator is a sequential create of input for large image training with custom function support, you can select the target directory and image decode transform by its defaults.

```
train_image_generator = ImageDataGenerator(rescale=1. / 255, vertical_flip=True, horizontal_flip=True, 
                preprocessing_function=custom_image_preprocess,) 
train_data_gen = train_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
	class_mode='binary',
    color_mode='rgb',
	seed=seed_1,)
	
test_image_generator = ImageDataGenerator(rescale=1. / 255, vertical_flip=True, horizontal_flip=True, 
              preprocessing_function=custom_image_preprocess,)
test_data_gen = test_image_generator.flow_from_directory(batch_size=BATCH_SIZE,
    directory=test_dir,
    shuffle=False,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
	class_mode='binary',
    color_mode='rgb',
	seed=seed_2,)
```

## Custom Function ##

The custom function to create random rotation as the questioner ask, it is not supported to rotate picture but for training, a process must have reasons to support it. One is rotating image is small and we do not notice with simple degrees that humans make mistakes.

```
def custom_image_preprocess( image ):

    random_lotation_layer = tf.keras.layers.RandomRotation(
        factor=(-0.2, 0.3),
        fill_mode='nearest',
        interpolation='nearest',
        seed=None,
        fill_value=0.0,
    )
	
    return random_lotation_layer(image)
```

## Medical perceptions result model ##

```
base_model = tf.keras.applications.Xception( weights='imagenet', 
			input_shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS), include_top=False)  
base_model.trainable = False
inputs = tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, IMG_CHANNELS))

x = tf.keras.applications.xception.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)
```

## Binary CrossEntropy and model training ##

```
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
		metrics=tf.keras.metrics.BinaryAccuracy())
model.fit(train_data_gen, epochs=10, validation_data=test_data_gen)
```

## References ##

1. https://stackoverflow.com/questions/74725957/indexerror-tuple-index-out-of-range-for-tensorflow-machine-learning-code
2. https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images?resource=download
3. https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator


## Training DATA ##

![MildDemented](https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/mildDem1.jpg "MildDemented") ![MildDemented](https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/nonDem40.jpg "MildDemented") ![MildDemented](
https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/moderateDem39.jpg "MildDemented") ![MildDemented](
https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/verymildDem63.jpg "MildDemented")



