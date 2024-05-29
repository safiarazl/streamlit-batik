from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from imgaug import augmenters as iaa
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import json

folder_path = 'model'
model_file = os.path.join(folder_path, 'model.h5')
print("model_file: ", model_file)
model = load_model(model_file)

# fungsi untuk mempersiapkan gambar sebelum diprediksi


def convert_image(image, IMG_SIZE=224):
    # membuat gambar menjadi grayscale
    image = image.convert('L')

    seq = iaa.Sequential([
        iaa.Resize({"width": IMG_SIZE, "height": IMG_SIZE}),
        iaa.PadToFixedSize(width=IMG_SIZE, height=IMG_SIZE,
                           position="center", pad_cval=255)
    ])

    # membuat gambar menjadi array
    image = seq.augment_image(img_to_array(image))

    # membuat array gambar tadi didalam array lagi, karena
    image = np.array([image])
    return image


def prediction_batik(image):
    plt.imshow(image)
    img = convert_image(image)

    # Load label/encoder
    # Baca file JSON
    with open('label/label.json') as f:
        data = json.load(f)

    # Simpan nilai dalam variabel
    label = data
    int_to_class = label['int_to_class']

    # Predict
    a = model.predict(img)
    predicted_class = str(int_to_class[str(np.argmax(a))])
    probability = a[0][np.argmax(a)]

    return predicted_class, probability


def main():
    # Load label/encoder
    # Baca file JSON
    with open('label/label.json') as f:
        data = json.load(f)

    # Simpan nilai dalam variabel
    label = data['class_to_int']
    class_to_int = list(label.keys())
    st.title("Image Batik Recognition")
    st.header("Kelas yang tersedia:")
    st.text(f"{class_to_int}")
    st.text("Upload an image to predict the batik.")

    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        predicted_class, probability = prediction_batik(image)
        st.header("Batik Prediction :")
        st.subheader(f"Predicted class: {predicted_class}")
        st.subheader(f"Probability: {probability:.2f}")


if __name__ == '__main__':
    main()
