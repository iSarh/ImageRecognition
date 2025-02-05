# Image Recognition Using TensorFlow

## Overview
This project demonstrates an image recognition system trained using **Teachable Machine by Google**. The trained model is exported in TensorFlow format and integrated into a Python script that loads the model, processes input images, and predicts their class.

## Features
- Trained a custom image recognition model with at least two classes.
- Model exported in TensorFlow **SavedModel** format.
- Python script for loading the model and making predictions on input images.
- Uses **PIL** for image processing and **NumPy** for numerical computations.

## Technologies Used
- **TensorFlow** (for loading and running the model)
- **PIL (Pillow)** (for image processing)
- **NumPy** (for handling image arrays)
- **Teachable Machine by Google** (for model training)


## Installation & Setup
### Prerequisites
Ensure you have **Python 3.11** installed along with the required dependencies:
```sh
pip install tensorflow pillow numpy
```

### Clone the Repository
```sh
git clone https://github.com/iSarh/ImageRecognition.git
cd ImageRecognition
```

### Running the Model
Run the script with a test image:
```sh
python main.py
```
The output will display the predicted class of the image.

## How It Works
1. Loads the trained model using TensorFlow's `saved_model.load()`.
2. Preprocesses an input image by resizing and normalizing it.
3. Pass the image through the model to get predictions.
4. Output the predicted class based on the model's confidence scores.

## Example Output
```
![Screenshot 2025-02-06 020937](https://github.com/user-attachments/assets/7b046e00-24e1-4d7d-811c-8b5e84a2efee)
![Screenshot 2025-02-06 023330](https://github.com/user-attachments/assets/c60d5318-eaa4-48a4-990b-2d01567e44a4)

```

## Model Training Steps
1. **Teachable Machine** was used to train a model with `sunflower` and `black-eyed-susan` classes.
2. The trained model was exported in **TensorFlow format**.
3. The exported model was integrated into a Python script for real-time predictions.

## Contributions
Feel free to contribute by submitting a pull request or opening an issue.

## License
This project is open-source and available under the **MIT License**.


