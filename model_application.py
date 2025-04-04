# this is for identifying live and dead single cells in a microwell
# Input:    a brightfield (BF) image and a merged image combines brightfield and fluorescent channels.
#           The brightfield image is the input of machine learning models
# (optional)The merged image provides the ground truth label to validate model performance  
# Output:
#        BF unlabeled foler: A folder contains all segmented nanowell images
#        Two folders contain the predicted and true single cell images, with one folder for each.
#        Two folders contain the predicted and true non-single cell images, with one folder for each.

#        Two folders contain the predicted and true live single images, with one folder for each.
#        Two folders contain the predicted and true dead single images, with one folder for each.


import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
from tensorflow.keras.models import load_model

# load brightfield and merged images
def load_2_imgs(base_folder,parent_name,well_name):
    load_imgs=[]
    name_list=['RGB.tif','BF_20X.tif']
    for i in range(len(name_list)):
        name=base_folder+parent_name+well_name+name_list[i]
        img=cv2.imread(name)
        if i >0:  # If it's not a merged image, convert it to grayscale
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        load_imgs.append(img)
    return load_imgs

# Normalize the brightness of the image
def normalize_brightness(image):
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val + 1e-9)  # Adding a small value to avoid division by zero
    return normalized_image

# model prediction on image data
def run_model(path_to_test_data,save_path0,save_path1,model,the):
    # load unsorted image data
    file_names=[f for f in os.listdir(path_to_test_data) if f.endswith('.jpg') ]

    for img_name in file_names:
        img_path=os.path.join(path_to_test_data,img_name)
        #load the img
        img=cv2.imread(img_path)
        height, width, channels = img.shape
        if height == 248 and width == 248 and channels == 3:
            # normalize the image
            normalized_image = normalize_brightness(img)
            # Expand dimensions 
            image_input = np.expand_dims(normalized_image, axis=[0, -1])
            img_pred = model.predict(image_input)
            if img_pred[0][0]>=the: shutil.copy2(img_path, os.path.join(save_path0, img_name))
            else: shutil.copy2(img_path, os.path.join(save_path1, img_name))

# find all nanowells in an image
def find_nanowell(BF_img,opening):
    # area range of nanowells
    min_area = 30000.0
    max_area = 70000.0
    nanowell_size=248
    offset_x=-10
    offset_y=-10
    # define the center region to find nanowells. Remove nanonwells at image edges
    xMin,xMax=200,8400 
    yMin,yMax=200,8400

    cnts = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    image_number = 0

    fig, ax = plt.subplots(figsize=(10, 10))
    centroidsBF=[]
    centroidsRGB=[]
    half_size = nanowell_size // 2
    for c in cnts:
        area = cv2.contourArea(c)
        if area > min_area and area < max_area:
            # compute the center of the contour
            M = cv2.moments(c)
            x = int(M["m10"] / M["m00"])+offset_x
            y = int(M["m01"] / M["m00"])+offset_y
            if xMin<x<xMax and yMin<y<yMax:
                centroidsBF.append((x,y))
                centroidsRGB.append((x,y))
                square = plt.Rectangle((x - half_size, y - half_size), nanowell_size, nanowell_size, 
                                    fill=False, edgecolor='red',linewidth=0.5)
                image_number+=1
                ax.add_patch(square)

    ax.imshow(BF_img,cmap='gray')
    print('total=',image_number)
    plt.show()
    return centroidsBF,centroidsRGB

# crop out the nanowells from raw images
def crop_squares(image, centroids, square_size, save_path):
    
    for i in range(len(centroids)):
        x, y = centroids[i]
        
        # Calculate the top-left corner of the square
        top_left_x = int(x - square_size / 2)
        top_left_y = int(y - square_size / 2)
        
        # Crop the square
        square = image[top_left_y:top_left_y+square_size, top_left_x:top_left_x+square_size]

        square_save=save_path+'_'+str(i)+'.jpg'
        cv2.imwrite(square_save,square)   


# load the model (1st CNN model for single cell identifcaiton)
model_path1= 'Y:/1stCNNdata/saved model/905_2/graham_Xception_val.h5'
model1=load_model(model_path1)
# load the model (2nd CNN model for live/dead assessment)
model_path2= 'Y:/2ndCNNdata/saved model/0907/graham_Xception_val.h5'
model2=load_model(model_path2)

# load the BF and merged images
# If no merged image, load BF only
base_folder= 'Z:/orig_data/PC3/Bor/'
parent_name='PC3_Bort_'
well_name='XY5_' # the name of this nanowell
load_imgs=load_2_imgs(base_folder,parent_name,well_name)

# find nanowells in BF img
BF_img=load_imgs[1]
# setting threshold of gray image 
ret, threshold = cv2.threshold(BF_img.copy(), 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
close = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (40,40))
opening = cv2.morphologyEx(close, cv2.MORPH_OPEN, kernel2)
# the centroids of each identified nanowell
centroidsBF,centroidsRGB=find_nanowell(BF_img,opening)


# Prepare the input for machine learning models
# save segmented nanowell images to a folder
base_save='Z:/test_data/PC3/Bor/0304/'
# crop nanowell images from the BF image
BFimg_folder=os.path.join(base_save,well_name,'BF unlabeled')
os.makedirs(BFimg_folder)
crop_squares(BF_img, centroidsBF, 248, os.path.join(BFimg_folder,well_name) )
# If there's a merged image:
# crop nanowells images from the merged image
RGBimg_fold=os.path.join(base_save,well_name,'merged','unsorted')
os.makedirs(RGBimg_fold)
crop_squares(load_imgs[0], centroidsRGB, 400, os.path.join(RGBimg_fold,well_name) )


# folders for true/predicted single and non-single cells, true/predicted live and dead cells
# manually calssify nanowell images and put them in their true folders
true_single=os.path.join(base_save,well_name,'merged','single')
true_non_single=os.path.join(base_save,well_name,'merged','non_single')
true_dead=os.path.join(base_save,well_name,'merged','single red')
true_live=os.path.join(base_save,well_name,'merged','single green')
pre_single=os.path.join(base_save,well_name,'predicted_single_cells')
pre_non_single=os.path.join(base_save,well_name,'predicted_non_single')
pre_live=os.path.join(base_save,well_name,'predicted_live')
pre_dead=os.path.join(base_save,well_name,'predicted_dead')

os.makedirs(true_single)
os.makedirs(true_non_single)
os.makedirs(true_dead)
os.makedirs(true_live)
os.makedirs(pre_single)
os.makedirs(pre_non_single)
os.makedirs(pre_live)
os.makedirs(pre_dead)

# 1st CNN
# single cell prediciton
run_model(BFimg_folder,pre_non_single,pre_single,model1,0.5)


# 2nd CNN 
# live/dead cell prediciton
run_model(pre_single,pre_dead,pre_live,model2,0.5)