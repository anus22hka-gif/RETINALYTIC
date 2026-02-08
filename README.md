# Rectanylatic ⭐

**Rectanylatic** is an end‑to‑end AI‑powered eye‑care analytics platform designed to assist in the early detection, analysis, and visualization of retinal conditions using medical imaging and machine learning. The system combines deep learning, computer vision, and an intuitive user interface to support clinicians, researchers, and healthcare innovators in making faster and more reliable retinal assessments.

---

## 🚀 Project Overview

Eye diseases such as **Diabetic Retinopathy, Glaucoma, and Age‑related Macular Degeneration (AMD)** are among the leading causes of preventable blindness worldwide. Early detection is critical, yet manual retinal analysis is time‑consuming and expertise‑dependent.

Rectanylatic addresses this challenge by providing:

* Automated retinal image analysis using AI/ML models
* Visual explanations and severity insights
* A scalable, modular, and clinically‑oriented system design

---

## 🎯 Key Objectives

* Enable **early detection** of retinal abnormalities
* Reduce diagnostic time with **AI‑assisted analysis**
* Provide **clear visual outputs** for better interpretability
* Build a **deployable, end‑to‑end healthcare AI system**

---

## 🧠 Core Features

* 📷 **Retinal Image Upload & Pre‑processing**
* 🤖 **Deep Learning–based Disease Detection**
* 📊 **Severity Classification & Confidence Scores**
* 🔍 **Region‑of‑Interest Highlighting (Explainable AI)**
* 🖥️ **User‑friendly Web Interface**
* 📁 **Modular and Scalable Architecture**

---

## 🏗️ System Architecture

1. **Data Layer**

   * Retinal fundus image datasets (public or custom)
   * Image normalization and augmentation

2. **Model Layer**

   * CNN / Transfer Learning models (ResNet, EfficientNet, etc.)
   * Trained for multi‑class retinal disease classification

3. **Inference & Analytics Layer**

   * Prediction engine
   * Severity scoring and confidence estimation

4. **Application Layer**

   * Web interface (Streamlit / Flask / FastAPI)
   * Visualization of predictions and insights

---

## 🧪 Dataset

Rectanylatic is trained on:

* **Kaggle Ocular Disease Recognition**


---

## 🛠️ Tech Stack

* **Programming Language:** Python
* **Machine Learning:** TensorFlow / PyTorch, Keras
* **Computer Vision:** OpenCV
* **Web Framework:** Streamlit / Flask
* **Data Handling:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Project Structure

```
Rectanylatic/
│
├── dataset/
│   └── ODIR-5K/
│       ├── Training Images/
│       ├── Testing Images/
│       └── data.xlsx
│
├── model/
│   ├── best_retina_model.h5
│   └── retina_odir_final.h5
│
├── utils/
│   ├── gradcam.py          # Explainable AI (Grad-CAM visualizations)
│   ├── preprocess.py       # Image preprocessing pipeline
│   └── report.py           # Automated medical-style report generation
│
├── train.py                # Model training script
├── predict.py              # Inference and prediction logic
├── app.py                  # Main Streamlit application
├── app2.py                 # Alternate / experimental UI flow
├── requirements.txt        # Project dependencies
├── README.md
└── .venv/                  # Virtual environment
```

---

## ▶️ How to Run the Project

1. **Clone the repository**

```bash
git clone https://github.com/your-username/rectanylatic.git
cd rectanylatic
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the application**

```bash
streamlit run app/app.py
```

---

## 📈 Results & Performance

* High accuracy on benchmark retinal datasets
* Robust performance on varying image quality
* Clear visual feedback for predictions

*(Exact metrics depend on dataset and training configuration.)*

---

## 👩‍💻 Author

**Anushka**
Healthcare AI Enthusiast
