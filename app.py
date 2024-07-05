# QCNN

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import os
import pennylane as qml

# Load the quantum model
filedir = 'files'
q_model = load_model(f'{filedir}/QModel.h5')

# Quantum circuit
wires = 4
dev4 = qml.device("default.qubit", wires=wires)

@qml.qnode(dev4)
def CONVCircuit(phi):
    theta = np.pi / 2
    qml.RX(phi[0] * np.pi, wires=0)
    qml.RX(phi[1] * np.pi, wires=1)
    qml.RX(phi[2] * np.pi, wires=2)
    qml.RX(phi[3] * np.pi, wires=3)
    qml.CRZ(theta, wires=[1, 0])
    qml.CRZ(theta, wires=[3, 2])
    qml.CRX(theta, wires=[1, 0])
    qml.CRX(theta, wires=[3, 2])
    qml.CRZ(theta, wires=[2, 0])
    qml.CRX(theta, wires=[2, 0])
    return qml.expval(qml.PauliZ(wires=0))

def QCONV1(X):
    H, W = X.shape
    step = 2
    out = np.zeros(((H // step), (W // step)))
    for i in range(0, W, step):
        for j in range(0, H, step):
            phi = X[i:i + 2, j:j + 2].flatten()
            measurement = CONVCircuit(phi)
            out[i // step, j // step] = measurement
    return out

def make_prediction(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 0)

    if img.shape[0] != 512 or img.shape[1] != 512:
        return "Error: Image should be 512x512"

    scale_percent = 25
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    NorImages = resized / 255.0
    processed = QCONV1(NorImages)
    images = np.asarray([processed])

    yhat = q_model.predict(images)
    yhat = yhat.argmax(axis=1)
    
    tumor_names = {
        1: "Meningioma",
        2: "Glioma",
        3: "Pituitary Tumor"
    }

    return tumor_names.get(int(yhat[0]), "Unknown")

def main():
    st.title("Brain Tumor Classifier using QCNN")

    # Patient details input fields
    name = st.text_input("Name:")
    dob = st.date_input("Date of Birth:")
    phone_number = st.text_input("Phone Number:")

    # Upload MRI/CT image
    st.write("\nUpload an MRI/CT image to classify the type of brain tumor.")
    uploaded_file = st.file_uploader("Choose an MRI/CT image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption='Uploaded MRI/CT Image', use_column_width=True)

        # Make predictions when button is clicked
        if st.button("Classify"):
            predicted_class = make_prediction(uploaded_file)
            
            if predicted_class.startswith("Error"):
                st.error(predicted_class)
            else:
                st.success(f"The MRI/CT image is classified as: {predicted_class}")

                # Save patient details and tumor name to Excel file
                data = {
                    'Name': [name],
                    'Date of Birth': [dob],
                    'Phone Number': [phone_number],
                    'Tumor Name': [predicted_class]
                }
                df = pd.DataFrame(data)
                file_path = "C:/Users/raksh/OneDrive/文档/Brain_tumor/patient_details.xlsx"

                try:
                    if not os.path.isfile(file_path):
                        df.to_excel(file_path, index=False, engine='openpyxl')
                    else:
                        existing_df = pd.read_excel(file_path, engine='openpyxl')
                        updated_df = pd.concat([existing_df, df], ignore_index=True)
                        updated_df.to_excel(file_path, index=False, engine='openpyxl')

                    st.success(f"Patient details saved to {file_path}")
                except Exception as e:
                    st.error(f"Error occurred while saving to Excel: {e}")

if __name__ == "__main__":
    main()


# DenseNet

# import streamlit as st
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# import cv2
# import os

# # Load DenseNet model
# model_path = 'DenseNet121.h5'
# model = load_model(model_path)

# # Function to preprocess image for model prediction
# def preprocess_image(img):
#     scale_percent = 25  # percent of original size
#     width = int(img.shape[1] * scale_percent / 100)
#     height = int(img.shape[0] * scale_percent / 100)
#     dim = (width, height)
#     resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
#     normalized_img = resized / 255.0  # Normalize image
#     return np.expand_dims(normalized_img, axis=0)

# # Function to predict image class
# def predict_image(image):
#     # Preprocess image
#     processed_image = preprocess_image(image)
#     # Predict using the model
#     prediction = model.predict(processed_image)
#     predicted_class = np.argmax(prediction)
#     return predicted_class

# def main():
#     st.title("Brain Tumor Classifier using DenseNet121")

#     # Patient details input fields
#     name = st.text_input("Name:")
#     dob = st.date_input("Date of Birth:")
#     phone_number = st.text_input("Phone Number:")

#     # Upload MRI/CT image
#     st.write("\nUpload an MRI/CT image to classify the type of brain tumor.")
#     uploaded_file = st.file_uploader("Choose an MRI/CT image...", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         # Display the uploaded image
#         st.image(uploaded_file, caption='Uploaded MRI/CT Image', use_column_width=True)

#         # Make predictions when button is clicked
#         if st.button("Classify"):
#             # Convert file to OpenCV format
#             file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#             img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

#             # Ensure the image is resized to match model input shape
#             if img.shape[0] < 512 or img.shape[1] < 512:
#                 st.error("Image size should be at least 512x512 pixels.")
#                 return

#             # Predict using DenseNet model
#             predicted_class = predict_image(img)

#             # Display the classification result
#             tumor_names = {
#                 1: "Meningioma",
#                 2: "Glioma",
#                 3: "Pituitary Tumor"
#             }
#             if predicted_class in tumor_names:
#                 st.success(f"The MRI/CT image is classified as: {tumor_names[predicted_class]}")
#             else:
#                 st.error("Unknown tumor type predicted.")

#             # Save patient details and classification result to Excel file
#             data = {
#                 'Name': [name],
#                 'Date of Birth': [dob],
#                 'Phone Number': [phone_number],
#                 'Tumor Name': [tumor_names.get(predicted_class, "Unknown")]
#             }
#             df = pd.DataFrame(data)
#             file_path = "C:/Users/raksh/OneDrive/文档/Brain_tumor/patient_details.xlsx"
#             df.to_excel(file_path, index=False)
#             st.success(f"Patient details and tumor classification saved to {file_path}")

# if __name__ == "__main__":
#     main()


# ResNet

# import streamlit as st
# import numpy as np
# import cv2
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.resnet_v2 import preprocess_input
# from sklearn.metrics import classification_report
# import pandas as pd
# import os

# def preprocess_image(uploaded_file):
#     # Read and preprocess the image
#     img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img, (128, 128))  # Resize to (128, 128) for compatibility with ResNet50V2
#     img = preprocess_input(img)
#     img = np.expand_dims(img, axis=0)
#     return img


# # Load the saved ResNet50V2 model
# model_path = 'ResNet50V2_custom.h5'
# model = load_model(model_path)

# # Streamlit app function
# def main():
#     st.title("Brain Tumor Classifier using ResNet50V2")
    
#     # Patient details input fields
#     name = st.text_input("Name:")
#     dob = st.date_input("Date of Birth:")
#     phone_number = st.text_input("Phone Number:")
    
#     # Upload MRI/CT image
#     st.write("\nUpload an MRI/CT image to classify the type of brain tumor.")
#     uploaded_file = st.file_uploader("Choose an MRI/CT image...", type=["jpg", "jpeg", "png"])
    
#     if uploaded_file is not None:
#         # Display the uploaded image
#         st.image(uploaded_file, caption='Uploaded MRI/CT Image', use_column_width=True)
        
#         # Make predictions when button is clicked
#         if st.button("Classify"):
#             # Preprocess the uploaded image
#             img = preprocess_image(uploaded_file)
            
#             # Perform prediction
#             predicted_probs = model.predict(img)
#             predicted_class = np.argmax(predicted_probs)
            
#             # Define tumor names corresponding to classes
#             tumor_names = {1: 'Meningioma', 2: 'Glioma', 3: 'Pituitary tumor'}
            
#             # Display the classification result
#             st.success(f"The MRI/CT image is classified as: {tumor_names[predicted_class]}")
            
#             # Save patient details and tumor name to Excel file
#             data = {
#                 'Name': [name],
#                 'Date of Birth': [dob],
#                 'Phone Number': [phone_number],
#                 'Tumor Name': [tumor_names[predicted_class]]
#             }
#             df = pd.DataFrame(data)
            
#             # Specify the file path for saving patient details
#             file_path = "C:/Users/raksh/OneDrive/文档/Brain_tumor/patient_details.xlsx"
#             df.to_excel(file_path, index=False)
#             st.success(f"Patient details saved to {file_path}")

# if __name__ == "__main__":
#     main()
