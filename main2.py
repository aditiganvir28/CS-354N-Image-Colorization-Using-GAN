import streamlit as st
import tensorflow as tf
import cv2
from tensorflow import keras
from io import BytesIO
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def downsample(filters, size, apply_batchnorm=True):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',kernel_initializer='he_normal', use_bias=False))
    if apply_batchnorm:
        result.add(tf.keras.layers.BatchNormalization())
    result.add(tf.keras.layers.LeakyReLU())
    return result

def upsample(filters, size, apply_dropout=False):
    result = tf.keras.Sequential()
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,padding='same',kernel_initializer='he_normal',use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())
    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())
    return result

def Generator():
    inputs = tf.keras.layers.Input(shape=[256,256,3])

    down_stack = [
        downsample(64, 4, apply_batchnorm=False), # (bs, 128, 128, 64)
        downsample(128, 4), # (bs, 64, 64, 128)
        downsample(256, 4), # (bs, 32, 32, 256)
        downsample(512, 4), # (bs, 16, 16, 512)
        downsample(512, 4), # (bs, 8, 8, 512)
        downsample(512, 4), # (bs, 4, 4, 512)
        downsample(512, 4), # (bs, 2, 2, 512)
        downsample(512, 4), # (bs, 1, 1, 512)
    ]

    up_stack = [
        upsample(512, 4, apply_dropout=True), # (bs, 2, 2, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 4, 4, 1024)
        upsample(512, 4, apply_dropout=True), # (bs, 8, 8, 1024)
        upsample(512, 4), # (bs, 16, 16, 1024)
        upsample(256, 4), # (bs, 32, 32, 512)
        upsample(128, 4), # (bs, 64, 64, 256)
        upsample(64, 4), # (bs, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose(3, 4,strides=2,padding='same',kernel_initializer=initializer,activation='tanh') # (bs, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)

# Create an instance of the generator model
generator = Generator()

SIZE = 256

img2 = None

# Load the weights into the model
generator.load_weights('modelGen2.h5')

# Define a function to generate color images
def generate_color_image(gray_image):
    global img2
    img2 = gray_image.astype("float32")
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # print(img2.shape)
    try:
        img2 = img2.reshape(1,SIZE,SIZE,3)
    except:
        img2 = cv2.resize(img2,(SIZE,SIZE))
        img2 = img2.reshape(1,SIZE,SIZE,3)

    img2 = img2.reshape(1, SIZE, SIZE, 3)
    # img2.shape
    # img2
    img2[0] = img2[0]/255.0
    # img2
    plt.imshow(img2[0])
    plt.savefig('output.png')
    plt.show()

    Pred=generator(img2, training=True)

    return Pred

# Streamlit app
def main():
    st.title("Colorization App")

    # Upload gray image
    uploaded_file = st.file_uploader("Upload a gray image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        gray_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Generate color image
        color_image = generate_color_image(gray_image)
       
        color_image_np = np.squeeze(color_image.numpy())

        # Create two columns
        col1, col2 = st.columns(2)

        # Display original image in the first column
        col1.subheader("Original Image")
        col1.image(img2, channels="RGB", use_column_width=0.5, clamp=True)

        # Display colorized image in the second column
        col2.subheader("Colorized Image")
        col2.image(color_image_np, channels="RGB", use_column_width=0.5, clamp=True)

if __name__ == '__main__':
    main()
