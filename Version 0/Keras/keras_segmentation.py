import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # change to CPU
from keras.models import Model,load_model
from keras.utils.np_utils import normalize
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, \
    Dropout, Lambda
import cv2
import numpy as np
from patchify import patchify, unpatchify
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from matplotlib import pyplot as plt

import time

# custom unet model
def encoder_block(input1, feature, kernel, dropout):
    c = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same')(input1)
    c = Dropout(dropout)(c)
    c = Conv2D(feature, (kernel, kernel), activation='relu', kernel_initializer='he_normal', padding='same')(c)
    return c
    
def decoder_block(input1, input2, feature, kernel, dropout):
    u = Conv2DTranspose(feature, (2, 2), strides=(2, 2), padding='same')(input1)
    u = concatenate([u, input2])
    c = encoder_block(u, feature, kernel, dropout)
    return c

################################################################
def unet_model(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, kernel):
    # Build the model
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    # s = Lambda(lambda x: x / 255)(inputs)   #No need for this if we normalize our inputs beforehand
    s = inputs

    # Contraction path
    c1 = encoder_block(s, 16, kernel, 0.1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = encoder_block(p1, 32, kernel, 0.1)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = encoder_block(p2, 64, kernel, 0.2)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = encoder_block(p3, 128, kernel, 0.2)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = encoder_block(p4, 256, kernel, 0.3)

    # Expansive path
    c6 = decoder_block(c5, c4, 128, kernel, 0.2)

    c7 = decoder_block(c6, c3, 64, kernel, 0.2)

    c8 = decoder_block(c7, c2, 32, kernel, 0.1)

    c9 = decoder_block(c8, c1, 16, kernel, 0.1)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=[inputs], outputs=[outputs])
    # optimizer = adam_v2.Adam(learning_rate=lr, decay=lr/epochs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    # model.summary()

    return model

# to mask the prediction image
def segmentation_map_to_image(
    result: np.ndarray, colormap: np.ndarray, remove_holes=False
) -> np.ndarray:

    if len(result.shape) != 2 and result.shape[0] != 1:
        raise ValueError(
            f"Expected result with shape (H,W) or (1,H,W), got result with shape {result.shape}"
        )

    if len(np.unique(result)) > colormap.shape[0]:
        raise ValueError(
            f"Expected max {colormap[0]} classes in result, got {len(np.unique(result))} "
            "different output values. Please make sure to convert the network output to "
            "pixel values before calling this function."
        )
    elif result.shape[0] == 1:
        result = result.squeeze(0)

    result = result.astype(np.uint8)

    contour_mode = cv2.RETR_EXTERNAL if remove_holes else cv2.RETR_TREE
    mask = np.zeros((result.shape[0], result.shape[1], 3), dtype=np.uint8)
    for label_index, color in enumerate(colormap):
        label_index_map = result == label_index
        label_index_map = label_index_map.astype(np.uint8) * 255
        contours, hierarchies = cv2.findContours(
            label_index_map, contour_mode, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            mask,
            contours,
            contourIdx=-1,
            color=color.tolist(),
            thickness=cv2.FILLED,
        )

    return mask

def segmentation(path):

    start = time.time()
    
    img_size = 512
    model = unet_model(img_size, img_size, 1, 3)
    model.load_weights('model_unet.hdf5')
    
    large_image = cv2.imread(path)
    lab_img = cv2.cvtColor(large_image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(l)
    updated_lab_img2 = cv2.merge((cl_img, a, b))
    bgr_image = cv2.cvtColor(updated_lab_img2, cv2.COLOR_LAB2BGR)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    large_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    #############################################################################
    # predict by patches
    # large image to small patches
    patches = patchify(large_image, (img_size, img_size), step=img_size)

    predicted_patches = []
    for i in range(patches.shape[0]):
        for j in range(patches.shape[1]):
            # print(i, j)

            single_patch = patches[i, j, :, :]
            single_patch_norm = np.expand_dims(normalize(np.array(single_patch), axis=1), 2)
            single_patch_input = np.expand_dims(single_patch_norm, 0)

            # Predict and threshold for values above 0.5 probability
            single_patch_prediction = (model.predict(single_patch_input)[0, :, :, 0] > 0.5).astype(np.uint8)
            predicted_patches.append(single_patch_prediction)

    predicted_patches = np.array(predicted_patches)
    predicted_patches_reshaped = np.reshape(predicted_patches, (patches.shape[0], patches.shape[1], img_size, img_size))

    reconstructed_image = unpatchify(predicted_patches_reshaped, large_image.shape)
    
    end = time.time()
    print('Time taken: {:.2f} seconds'.format(end - start))
    #############################################################################
    # confluency calculation
    num_white = np.sum(reconstructed_image == 1)
    num_black = np.sum(reconstructed_image == 0)
    confluency = (num_white/(num_white+num_black))*100
    # print('Confluency: {}'.format(confluency))
    
    #############################################################################
    # segmentation masking
    height, width = reconstructed_image.shape
    
    # Define colormap, each color represents a class
    colormap = np.array([[0, 0, 0], [0, 255, 0]])
    
    # Define the transparency of the segmentation mask on the photo
    alpha = 0.3
    
    # Use function from notebook_utils.py to transform mask to an RGB image
    mask = segmentation_map_to_image(reconstructed_image, colormap)
    
    resized_mask = cv2.resize(mask, (width, height))
    
    # Create image with mask put on
    image_with_mask = cv2.addWeighted(bgr_image, 1-alpha, resized_mask, alpha, 0)
    image_with_mask = cv2.cvtColor(image_with_mask, cv2.COLOR_BGR2RGB)
    
    
    plt.figure(figsize=(15, 10))
    plt.subplot(231)
    plt.title('Original Image')
    plt.imshow(rgb_image)
    plt.subplot(232)
    plt.title('Prediction Mask')
    plt.imshow(reconstructed_image, cmap='gray')
    plt.subplot(233)
    plt.title('Masked Image')
    plt.imshow(image_with_mask)
    
    plt.show()
    
    # cv2.imwrite('image_with_mask.jpg', image_with_mask)

root = Tk()
root.withdraw()
root.lift()
root.attributes('-topmost',True)
root.after_idle(Tk().attributes,'-topmost',False)
path = askopenfilename(title='Choose test image', \
                       initialdir='D:/PycharmProjects/Cell Segmentation/')
    
segmentation(path)
root.quit()