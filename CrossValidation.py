# preparation: 
# split the total dataset into training (80%) and testing (20%)

# For the training dataset, create a CVS file ("img_labels.csv") to label each image
# (column1: image_name; column2: Lable (0 or 1 to represent live/dead or single/non-single))

# Put all training images into one folder (folder name: "train_cv")

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.models import load_model


def normalize_brightness(image):
    # Normalize the brightness of the image
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val + 1e-9)  # Adding a small value to avoid division by zero
    return normalized_image

# date
date='202XXXX'
# base_folder with all training and testing and dataframe
base_folder='/project/CNN/data/'
# the Xception model downloaded from Keras
model_path='/project/tf_xception_model/'
# Specify the directory where you want to save the file (data for confusion matrix)
save_pathParent = '/project/CNN/save_cv'
save_path=os.path.join(save_pathParent,date)
os.makedirs(save_path)

dataframe_path=os.path.join(base_folder,'img_labels.csv')
train_path=os.path.join(base_folder,'train_cv')
test_path=os.path.join(base_folder,'test')

IMG_SIZE = 248
BATCH_SIZE = 15
N_SPLIT = 5
EPOCHS = 15
# Model-----------------------------------------------------
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
# ----------------------------------------------------------------

data_kfold=pd.read_csv(dataframe_path)
train_label = data_kfold.Label
train_x = data_kfold.drop(['Label'],axis=1)

train_datagen = ImageDataGenerator(
        preprocessing_function=normalize_brightness,
        horizontal_flip=True,
        vertical_flip=True)
validation_datagen = ImageDataGenerator(preprocessing_function=normalize_brightness)

test_datagen = ImageDataGenerator(preprocessing_function=normalize_brightness)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',  
    shuffle=False  
)

cv = KFold(n_splits=N_SPLIT, shuffle=True, random_state=42)
fold_no = 0
for train_idx, val_idx in list(cv.split(train_x,train_label)):
    fold_no+=1
    print('   ')
    print(f'Training for fold {fold_no} ...')

    x_train_df=data_kfold.iloc[train_idx]
    x_val_df=data_kfold.iloc[val_idx]

    train_generator=train_datagen.flow_from_dataframe(
        dataframe=x_train_df,
        directory=train_path,
        x_col="Image_name",
        y_col="Label",
        class_mode="categorical",
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE
    )

    valid_generator=validation_datagen.flow_from_dataframe(
        dataframe=x_val_df,
        directory=train_path,
        x_col="Image_name",
        y_col="Label",
        class_mode="categorical",
        target_size=(IMG_SIZE,IMG_SIZE),
        batch_size=BATCH_SIZE        
    )

    history=model_Xception.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=EPOCHS,
        steps_per_epoch=x_train_df.shape[0]//BATCH_SIZE,
        verbose=1
    )

    # make predictions on unseen data using Trained model
    y_pred = model_Xception.predict(test_generator, steps=len(test_generator), verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    # Get true labels from the generator
    y_true = test_generator.classes

    # save each round of result into csv
    df_round=pd.DataFrame({'prediction': y_pred_classes,
                          'true_label': y_true})
    round_output=os.path.join(save_path,str(fold_no)+'fold.csv')
    df_round.to_csv(round_output,index=False) 



