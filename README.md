
<h1 align="center">🌼 Linear Neural Network for Image Classification</h1>

<p align="center">
  <b>A minimal, educational TensorFlow project demonstrating how a purely linear neural network can classify flower images.</b>
</p>

<p align="center">
  <a href="https://colab.research.google.com/drive/1jxgHkt9RB3R4WHNbEBgrj_sKF2qCgZfK" target="_blank">
    <img src="https://img.shields.io/badge/Run%20in%20Colab-F9AB00?logo=googlecolab&logoColor=white" alt="Run in Colab">
  </a>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow 2.x">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python 3.9+">
  <img src="https://img.shields.io/badge/License-Educational-green" alt="License: Educational">
</p>

---

## 📘 Overview

This project builds and trains a **Linear Neural Network (LNN)** to classify flower images into 5 categories:

["daisy", "dandelion", "roses", "sunflowers", "tulips"]


The model uses only:
- A **Flatten layer** to turn images into a single vector  
- A **Dense layer** with Softmax activation to output class probabilities  

No hidden layers, no convolutions — just pure linear learning.  
Ideal for understanding how neural networks process images at the most basic level.

---

## 🧠 Model Architecture

| Layer | Type | Output Shape | Description |
|--------|------|---------------|--------------|
| 1 | `Flatten` | (150,528) | Converts 224×224×3 image into 1D vector |
| 2 | `Dense` | (5) | Outputs probability for each flower class |

**Mathematical Representation:**


Y = {softmax}(B + W * X)

---

## 📂 Dataset

**Source:** [TensorFlow Flower Photos](https://www.tensorflow.org/datasets/catalog/tf_flowers)  
Accessed via Google Cloud Storage:
gs://cloud-ml-data/img/flower_photos/

Each CSV file lists image paths and labels:
image_path,label
gs://cloud-ml-data/img/flower_photos/daisy/001.jpg,daisy

---

## ⚙️ Setup

**Install dependencies:**
```bash
pip install tensorflow matplotlib numpy
```

🚀 Training

Hyperparameters

batch_size = 16
epochs = 2
optimizer = 'adam'
loss = SparseCategoricalCrossentropy(from_logits=False)


Train command

history = model.fit(
    train_dataset,
    validation_data=eval_dataset,
    epochs=epochs,
)


During training, both accuracy and loss are logged and plotted.

🧩 Key Learning Points

* How image tensors are preprocessed and normalized

* Building TensorFlow data pipelines from CSVs

* Flattening image tensors for dense network input

* Implementing Softmax for multi-class output

* Visualizing model predictions and training metrics

🔮 Future Work

* Add hidden layers with nonlinear activations (ReLU)

* Integrate Convolutional Neural Networks (CNNs)

* Apply Data Augmentation

* Experiment with transfer learning (e.g., MobileNetV2)

## 💡 Author

**👨‍💻 Raunak Srivastava**  
Built as part of a **Deep Learning project** exploring the fundamentals of image classification and neural network architectures.
