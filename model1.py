import tensorflow as tf
import pandas as pd
from keras.models import load_model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def crop_rgb(img_path):

    # Load the image Resize the image to 0.1 times the original size using nearest neighbour interpolation
    img = plt.imread(img_path)
    # img = cv2.resize(img, (0, 0), fx=0.1, fy=0.1, interpolation=cv2.INTER_NEAREST)
    
    # Initialize crop boundaries to cover the entire image
    top, bottom, left, right = 0, img.shape[0], 0, img.shape[1]
    centre = img.shape[0] // 2, img.shape[1] // 2

    left_shift, right_shift = 0, 0

    # Find the leftmost non-black column
    for i in range(centre[1], -1, -1):
        if np.sum(img[centre[0], i, :]) < 10:  # Assuming non-black pixels have a sum greater than 10
            left_shift = centre[1] - i
            break

    # Find the rightmost non-black column
    for i in range(centre[1], img.shape[1]):
        if np.sum(img[centre[0], i, :]) < 10:
            right_shift = i - centre[1]
            break

    # Update the crop boundaries
    shift= max(left_shift, right_shift)
    left = centre[1] - shift
    right = centre[1] + shift

    # Crop in a 3:5 ratio
    width = right - left
    new_height = int(width * 0.60)
    centre = img.shape[0] // 2
    top = max(0, centre - (new_height // 2))
    bottom = min(img.shape[0], centre + (new_height // 2))

    # Crop the image
    cropped_img = img[top:bottom, left:right, :]

    # return cropped image
    return cropped_img

def load_and_preprocess_image(image_path, image_size=(120, 200)):
    """
    Load an image file and preprocess it for model prediction.
    """
    
    data = pd.DataFrame({'File Path': [image_path], 'Final Label': ['RG']})
    img= crop_rgb(data['File Path'][0])
    # convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    cv2.imwrite('temp.jpg', gray_img)

    data = pd.DataFrame({'File Path': ['temp.jpg'], 'Final Label': ['NRG']})

    # Load the image using keras
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
    )
    generator = datagen.flow_from_dataframe(
        data,
        x_col='File Path',
        y_col='Final Label',
        target_size=image_size,
        class_mode='raw',
        batch_size=1,
        shuffle=True
    )
    
    # Return the preprocessed image
    return next(generator)[0]

def predict_image(model_path, image_path):
    # print('Image Path:', image_path)
    # Predict the class of an image using a trained model.

    # Load the pre-trained model
    model = load_model(model_path)
    
    # Load and preprocess the image
    try:
        preprocessed_image = load_and_preprocess_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        return 0.0
    
    # Predict the image
    prediction = model.predict(preprocessed_image)
    # print("prediction:\n", prediction)
    
    # Output the likelihood of the image belonging to the positive class
    likelihood = prediction[0][0]
    
    return likelihood

# Example usage
# model1_path = 'task1.h5'
# image_path = 'TRAIN000034.JPG'

# likelihood1 = predict_image(model1_path, image_path)
# print(f"The likelihood of the image being positive is: {likelihood1:.4f}")

