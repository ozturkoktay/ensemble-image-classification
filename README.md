# Cats vs Dogs Classification with Deep Learning

## Project Overview
This project classifies images of cats and dogs using transfer learning with three different deep learning models: MobileNetV2, InceptionV3, and Xception. Additionally, an ensemble model is created to combine the predictions of these models for improved accuracy.

## Dataset
The dataset used is the "Cats and Dogs Filtered" dataset from TensorFlow’s ML EDU datasets, automatically downloaded from Google Cloud Storage.

- **Training Data**: `train/cats` and `train/dogs`
- **Validation Data**: `validation/cats` and `validation/dogs`

## Environment Setup
### Requirements
Ensure you have the following dependencies installed:
```bash
pip install tensorflow numpy matplotlib opencv-python
```

## Model Architecture
This project utilizes pre-trained deep learning models for feature extraction:
- **MobileNetV2**
- **InceptionV3**
- **Xception**

Each of these models is fine-tuned for binary classification (cat vs dog), and their predictions are later combined using an ensemble model.

## Data Preprocessing
- Images are resized to **224x224** pixels.
- Normalization is applied to scale pixel values.
- Labels are encoded as `0` for **dogs** and `1` for **cats**.

## Training Process
### Steps:
1. Each base model (MobileNetV2, InceptionV3, Xception) is trained separately.
2. The trained models are saved as `.h5` files.
3. An ensemble model is created by combining the outputs of the three models.
4. The ensemble model is trained on the combined predictions.
5. Performance metrics are recorded and compared.

## Evaluation Metrics
The models are evaluated based on:
- **Accuracy** (Validation Accuracy)
- **Loss** (Binary Crossentropy)

Final Validation Accuracy:
```
MobileNetV2 acc: 0.9788
InceptionV3 acc: 0.9778
Xception acc: 0.9788
Ensemble acc: 0.9829
```

## Usage
To use the trained ensemble model for inference:
```python
import tensorflow as tf
import numpy as np
import cv2

def predict_image(image_path, model):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return 'Dog' if prediction < 0.5 else 'Cat'

model = tf.keras.models.load_model('models/model.h5')
print(predict_image('test_image.jpg', model))
```

## Results
- Individual models achieved **~97.8%** validation accuracy.
- The ensemble model improved performance to **98.3%** validation accuracy.

## Authors
This project was built using TensorFlow and Keras for deep learning-based image classification.

## License
This project is open-source and licensed under the MIT License.

