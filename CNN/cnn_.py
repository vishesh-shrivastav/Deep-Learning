# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()
# Step 1 - Add first layer - Convolutional layer
# 32 feature maps of 3*3 dimension
classifier.add(Conv2D(32, 3, 3, input_shape = (64, 64, 3), activation = "relu"))
# Step 2 - MaxPooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

"""
Improving the CNN
Add one more convolutional layer
No input_shape parameter required since input is pooled layer
Uncomment following lines to add extra layer
"""
#classifier.add(Conv2D(32, 3, 3, activation = "relu"))
#classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
# Step 4 - Full Connection
classifier.add(Dense(output_dim = 128, activation = "relu"))
classifier.add(Dense(output_dim = 1, activation = "sigmoid"))
# Compile the CNN
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

# Part 2 - Fitting the CNN to the images

# Image augmentation to prevent overfitting
# Generates many more images by random transformations of existing images
# Helps in making the training set bigger and better
# Enrich dataset without actually adding more images
# Gets good performance results with little or no overfitting

from keras.preprocessing.image import ImageDataGenerator
# Prepare train and test Image data generators
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

import os
os.getcwd()

# Generate training and test sets
training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

# Fit classifier on test images
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000, # number of images sampled in one epoch
                         epochs = 5, # Set epochs to 5 because CPU
                         validation_data = test_set,
                         validation_steps = 2000)

# Accuracy ~~ 75% on test set

# Part 3 - Making predictions with CNN
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64,64))

# Convert image to 3D array since our CNN uses 3*64*64 arrays as imput images
test_image = image.img_to_array(test_image)

# Add extra dimension since predict method expects four dimensions
test_image = np.expand_dims(test_image, axis = 0)

# Predict
result = classifier.predict(test_image) # result = 1
# Verify if prediction is correct
training_set.class_indices # {'cats': 0, 'dogs': 1}
# correct prediction!

test_image_2 = image.load_img('dataset/single_prediction/cat_or_dog_2.jpg', target_size = (64,64))
# Convert image to 3D array since our CNN uses 3*64*64 arrays as imput images
test_image_2 = image.img_to_array(test_image_2)

# Add extra dimension since predict method expects four dimensions
test_image_2 = np.expand_dims(test_image_2, axis = 0)

# Predict
result_2 = classifier.predict(test_image_2) # result_2 = 1, incorrect prediction

# Improving the model
