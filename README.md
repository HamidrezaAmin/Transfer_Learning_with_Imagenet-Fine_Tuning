# Transfer_Learning_with_Imagenet + Fine_Tuning

This project demonstrates how to use **Transfer Learning** with the **ImageNet pre-trained VGG16 model** and **fine-tune** it for a custom image classification task. It is implemented using TensorFlow and Keras in **Google Colab**.

---

## 🧠 Project Overview

We use the VGG16 model trained on ImageNet as a base and:
1. Add custom Dense layers on top.
2. Freeze early layers of VGG16 (to retain learned features).
3. Fine-tune later convolutional blocks to adapt to the new dataset.
4. Perform training with and without data augmentation.

---

## 🛠 Environment

- Python 3.x  
- TensorFlow 2.x  
- Keras (bundled with TensorFlow)  
- Google Colab  

---

## 📁 Dataset Structure

The dataset directory is expected to be structured as:

```
dataset/
├── train/
│   ├── class_1/
│   └── class_2/
├── validation/
│   ├── class_1/
│   └── class_2/
└── test/
    └── [Unlabeled Images]
```

---

## 📌 Steps to Run

### 1. Clone or Upload Your Dataset
Upload your dataset into your Colab environment or Google Drive.

### 2. Load Pretrained Model

```python
from keras.applications import VGG16
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
```

### 3. Freeze Base Layers (Transfer Learning)

```python
conv_base.trainable = False
```

### 4. Add Custom Layers

```python
from keras import models, layers
model = models.Sequential([
    conv_base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
```

### 5. Compile and Train

```python
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(learning_rate=1e-4),
              metrics=['acc'])

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator)
```

### 6. Fine-Tuning

```python
conv_base.trainable = True
set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    layer.trainable = set_trainable
```

Re-compile with a lower learning rate and retrain.

---

## ✅ Final Evaluation

Use `model.evaluate()` on test set (prepared as unlabeled images) to check performance.

---

## 📌 Notes

- Data augmentation was used to reduce overfitting.
- Functional API can be used for more flexibility (not covered here).
- Visualization of accuracy/loss helps interpret training performance.

---

## 📍 Reference

This project was built in Google Colab using TensorFlow's Keras API with support from ImageNet and VGG16 pretrained weights.
