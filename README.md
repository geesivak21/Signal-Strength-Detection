# **Signal Strength Classification Using Neural Networks**

### **Domain:** Electronics & Telecommunication

### **Project Type:** Machine Learning – Multiclass Classification

### **Objective:** Predict signal strength/quality emitted by communication equipment using measurable signal parameters.

---

## **📌 Project Overview**

A communications equipment manufacturing company performs multiple signal parameter tests on their devices. The company wants an automated solution to predict the **signal strength** using these parameters.

This project builds a **Neural Network–based classifier** that learns patterns from historical data and predicts the signal quality class.

---

## **📁 Dataset Description**

The dataset (`Signals.csv`) contains:

* **11 signal parameters** — numerical measurements recorded during signal tests
* **Signal_Strength** — target variable with classes (3, 4, 5, 6, 7, 8)

---

## **🧭 Steps Followed**

---

## **1. Data Import & Understanding**

### ✔ A. Import required libraries and load the dataset

* Loaded CSV as a pandas DataFrame
* Imported NumPy, Matplotlib, Seaborn, sklearn, and TensorFlow/Keras

### ✔ B. Missing values check

* Calculated percentage of missing values per column
* **Result:** No missing values found

### ✔ C. Duplicate check

* Identified duplicate records: **240 duplicates**
* Duplicates were removed/handled appropriately

### ✔ D. Target variable visualization

* Plotted distribution of **Signal_Strength** using histogram
* Class 5 had the highest count, class 3 the lowest

### ✔ E. Key insights

* Dataset is clean (no nulls) but contained duplicates
* Target classes are imbalanced
* Features appear normally distributed after scaling

---

## **2. Data Preprocessing**

### ✔ A. Separated features (X) and target (y)

### ✔ B. Train-test split

* 70% training, 30% testing

### ✔ C. Printed shapes to verify split

* Training set: 1119 samples
* Testing set: 480 samples

### ✔ D. Normalization

* Applied **StandardScaler** to X_train and X_test

### ✔ E. Label transformation

* Mapped class labels to integers
* Converted to **one-hot encoded vectors** for Neural Network training

---

## **3. Model Training & Evaluation (Neural Network)**

### ✔ A. Designed initial neural network

* 3 hidden layers (64 → 32 → 16 units)
* ReLU activation
* Softmax output
* Weight initialization: He uniform

### ✔ B. Model training

* Trained for 20 epochs
* Callbacks used:

  * ModelCheckpoint
  * ReduceLROnPlateau
  * EarlyStopping

### ✔ C. Plotted results

* **Training vs Validation Loss**
* **Training vs Validation Accuracy**

### ✔ D. Improved architecture

* Added more neurons (256 → 128)
* Added Batch Normalization + Dropout
* Retrained with same preprocessing

### ✔ E. Evaluation and insights

* The new architecture showed:

  * Better convergence
  * Lower validation loss
  * Higher stable accuracy
  * Reduced overfitting due to BatchNorm + Dropout

---

## **📌 Final Output**

* Saved trained model as:
  **`final_model.keras`**
* Also stored weights via checkpoints:

  * `model.weights.h5`
  * `model_updated.weights.h5`

---

## **📦 Tools & Technologies**

* Python
* NumPy, Pandas
* Matplotlib, Seaborn
* scikit-learn
* TensorFlow / Keras
* Google Colab

---

## **📈 Conclusion**

A Neural Network classifier was successfully developed to predict signal quality from various signal parameters.
The improved deep learning model demonstrated better performance and generalization.

This model can be integrated into the company’s systems to assist with automated signal quality prediction.
