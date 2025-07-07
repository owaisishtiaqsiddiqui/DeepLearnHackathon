# QML Challenge

<div align = "center">
  <img width='auto' height=300 src="https://blogger.googleusercontent.com/img/b/R29vZ2xl/AVvXsEjvFhfnoZf8qGq5B77keDh_bmSfdpSsd_MJXehTGKQOkfl4uUgBZ3fM8d0kjjcCGDfYDXwYzPpZf5moACKyK2Ejew-ldNRvAofzhHQXGTRYmJgHatvbLTR1nqXotI-QZj2sNVao87w5B6g/s1600/quantum+model.png">
  <img width='auto' height=300 src="https://prefetch.eu/know/concept/bloch-sphere/sketch-full.png?v=1">
</div>

The goal of the challenge is to use a quantum machine learning model to perform classification on the Higgs Boson dataset. The original dataset is available at [UCI Machine learning Repository](https://archive.ics.uci.edu/dataset/280/higgs) containing 11 million labelled samples. Afterwards, we want to use the model to classify Higgs vs background events using a test dataset.


## Dataset

For this challenge, we use only a subset of the dataset (5000 samples) provided in the `data` directory. Each data point consists of 28 features. The first 21 features are kinematic properties measured at the level of the detectors. The last seven are functions of the first 21. We later test the model on another set of 5000 samples to evaluate the submissions.

## Model

You are free to choose any model you like. However, in the spirit of quantum machine learning, we put a limit on the maximum number of parameters you are allowed to use.

## Challenge and Evaluation
Fill out the pre- and post-hackathon surveys

Submissions are evaluated based on **ROC AUC** on a hidden test set.

### Submission Requirements

- Two executable Jupyter Notebooks and their PDF versions, plus trained model weights.
- Only use PennyLane and the libraries available on Google Colab.

**Notebook 1:** Trains the model on the provided dataset and plots the final ROC Curve.

**Notebook 2:** Loads trained weights and plots the final ROC Curve. Define a `data_path` variable (to be set by moderators during testing).

### Submission Criteria

1. Use only the provided dataset subset for training (preprocessing allowed).
2. Quantum circuits must use ≤ 20 qubits. Hybrid models allowed, but prioritize quantum methods.
3. Evaluation metric: ROC AUC score.
4. Do not modify the final code cell that plots the ROC Curve (from `QML Challenge.ipynb`), as it's used for grading

## Example notebook

We provide few example notebooks to get started.

1. Hands-on Quantum Computing with PennyLane [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ML4SCI/DeepLearnHackathon/blob/main/QMLChallenge/Quantum_Computing_Warmup.ipynb)
2. Using a Quantum Neural Network (QNN) for binary classification on a Toy dataset (Iris flower data set) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ML4SCI/DeepLearnHackathon/blob/main/QMLChallenge/Quantum_Computing_Warmup%20-%202.ipynb).
3. Using a Classical Neural Network for binary classification on Higgs Boson Dataset. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ML4SCI/DeepLearnHackathon/blob/main/QMLChallenge/QML%20Challenge.ipynb).

**Note:** `QML Challenge.ipynb` uses a classical neural network but the submission must use a QNN.


To learn more about QML, checkout the official [demos](https://pennylane.ai/qml/demonstrations/) by PennyLane.

image courtesy: [tensorflow](https://blog.tensorflow.org/2020/03/announcing-tensorflow-quantum-open.html), [prefetch](https://prefetch.eu/know/concept/bloch-sphere/).

## Tips for working with Google Colaboratory

You can upload the dataset on your Google Drive and to use it on Google Colab. To use the dataset with Google Drive in Google Colab, follow the steps below:

- After opening the dataset link, click "Add shortcut to drive".
- Click the "All locations" tab and add it to "My Drive".

You can use the following code to mount Google Drive:

```
from google.colab import drive
drive.mount('/content/drive')
```

## Contributors

- Sergei V. Gleyzer *(University of Alabama)*
- Jogi Suda Neto *(University of Alabama)*
- Marco Knipfer *(University of Alabama)*
- Gopal Ramesh Dahale *(EPFL)*
- Ali Hariri *(EPFL)*
- Eric Reinhardt *(University of Alabama)*
