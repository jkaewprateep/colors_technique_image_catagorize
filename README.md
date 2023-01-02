# colors_technique_image_catagorize
For study color technique image catagorize, binary image catagorize.

## DataSet ##

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

## References ##

1. https://stackoverflow.com/questions/74725957/indexerror-tuple-index-out-of-range-for-tensorflow-machine-learning-code
2. https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images?resource=download


## Training DATA ##

![MildDemented](https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/mildDem1.jpg "MildDemented") ![MildDemented](https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/nonDem40.jpg "MildDemented") ![MildDemented](
https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/moderateDem39.jpg "MildDemented") ![MildDemented](
https://github.com/jkaewprateep/colors_technique_image_catagorize/blob/main/verymildDem63.jpg "MildDemented")



