import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
model = tf.saved_model.load(r"C:\Users\Sarah\PycharmProjects\ImageRecognition\model.savedmodel") # Ensure the correct relative or absolute path to the model


# Function to load and prepare the image
def load_and_prepare_image(image_path, target_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


# Function to predict the image class
def predict_image_class(image_path):
    image = load_and_prepare_image(image_path)

    # Get prediction function from model
    infer = model.signatures['serving_default']
    predictions = infer(tf.convert_to_tensor(image, dtype=tf.float32))

    # Print available keys to identify the correct output layer name
   # print("Available keys:", predictions.keys())

    # Use the correct key based on the printed keys
   # output_layer_name = list(predictions.keys())[0]  # Automatically get the first output key
   # print(f"Using output layer: {output_layer_name}")

    predicted_class = np.argmax(predictions['sequential_19'].numpy(), axis=1)
    return predicted_class[0]


# Test the program
if __name__ == "__main__":
    image_path = r"C:\Users\Sarah\PycharmProjects\ImageRecognition\testImage.jpg" # Ensure the correct relative or absolute path to the image
    class_index = predict_image_class(image_path)
    classes = ["sunflower", "black-eyed-susan"]  # Replace with your actual class names
    print(f"The image belongs to the class: {classes[class_index]}")
