import os
import pickle
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model, load_model
from keras.callbacks import CSVLogger
   
# Normalize the brightness of the image
def normalize_brightness(image):
    
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val + 1e-9)  # Adding a small value to avoid division by zero
    return normalized_image

# training data folder
folder_train='/project/CNN/data/train/'

batch_size = 15    
img_datagen = ImageDataGenerator(
        preprocessing_function=normalize_brightness,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.20)

train_generator = img_datagen.flow_from_directory(
    directory=folder_train,
    target_size=(248, 248),
    batch_size=batch_size,
    class_mode="categorical",
    subset='training',
    shuffle=True,
    seed=42,
)

valid_generator = img_datagen.flow_from_directory(
    directory=folder_train,
    target_size=(248, 248),
    batch_size=batch_size,
    class_mode="categorical",
    subset='validation',
    shuffle=False,
    seed=42,
)

# Define the input shape---------------------------------
input_shape = (248, 248, 3)  # rgb img has 3 channels
# Create a new input layer for grayscale images
input_layer = Input(shape=input_shape)
# Instantiate the Xception model
model_path='/project/tf_xception_model/'
base_model=load_model(model_path)

# Add a global spatial average pooling layer
x = base_model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
# Add a fully-connected layer
x = tf.keras.layers.Dense(1024, activation='relu')(x)
# Add a logistic layer for 2 classes
predictions = tf.keras.layers.Dense(2, activation='softmax')(x)
# This is the model we will train
model_Xception = Model(inputs=base_model.input, outputs=predictions)
model_Xception.compile(optimizer= Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
# Set up CSVLogger to log training progress
csv_logger = CSVLogger('training_log.csv', separator=',', append=False)


# train the model
BATCH_SIZE=32
total_train=train_generator.n
total_val=valid_generator.n
steps_per_epoch = int(np.ceil(total_train / float(BATCH_SIZE)))
history = model_Xception.fit(train_generator,
                    validation_data=valid_generator,
                    verbose=1,
                    steps_per_epoch=steps_per_epoch,     
                    epochs=15,
                    callbacks=[csv_logger])



# Specify the directory where you want to save the file
directory = '/project/CNN/save_model/20XXX'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.makedirs(directory)

# Specify the filename
filename1 = os.path.join(directory, 'Xception_val.pkl')
filename2 = os.path.join(directory, 'Xception_val.h5')

# Check if the file already exists
if not os.path.exists(filename2):
    model_Xception.save(filename2)
    # Save the history object to a file
    with open(filename1, 'wb') as f:
        pickle.dump(history.history, f)
else:
    filename2=os.path.join(directory, 'overlapped.h5')
    model_Xception.save(filename2)
