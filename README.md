Marvel Character Image Classifier  
*A Cinematic Deep Learning Experience built with PyTorch and Streamlit*

---

## 📖 Overview
This project is a **Marvel Character Image Classification System** that predicts which Marvel superhero appears in an input image.  
It uses **deep convolutional neural networks (ResNet-18)** to learn visual patterns from a curated Marvel dataset and provides a **cinematic prediction interface** built with Streamlit.

---

## 🎯 Problem Statement
The goal of this project is to develop a **baseline image classification system** capable of correctly identifying the class of a given image.  
For this case study, the chosen domain is **Marvel superheroes**, leveraging computer vision to recognize iconic characters from the Marvel universe.

---

## 📊 Dataset
The dataset was sourced from [Kaggle’s Marvel Character Image Classification Dataset](https://www.kaggle.com/code/gcdatkin/marvel-character-image-classification).  

- **Classes Used:** Iron Man, Captain America, Spider-Man, Thor  
- **Data Split:**  
  - Training: 70%  
  - Validation: 20%  
  - Testing: 10%  
- **Input Resolution:** 224×224  
- **Preprocessing:** Normalization, resizing, and random augmentations for generalization.

---

## 🧠 Models Trained
Several CNN architectures were explored to evaluate performance:

| Model | Parameters (M) | Accuracy | Remarks |
|:------|:---------------:|:--------:|:--------|
| Custom CNN | 1.2 | 78% | Light model, prone to overfitting |
| VGG-16 | 134 | 87% | Deep but slow |
| **ResNet-18** | 11 | **93%** | Best trade-off between accuracy and speed |

✅ **Final Model:** ResNet-18 (Fine-tuned on Marvel dataset)  
✅ **Optimizer:** Adam (lr=0.001)  
✅ **Loss Function:** Cross-Entropy Loss  
✅ **Batch Size:** 32  
✅ **Epochs:** 25  

---

## ⚙️ Training Summary
Training was performed in **PyTorch**, with the model saved as `models/marvel_resnet18.pth` containing:
- Trained weights (`model_state_dict`)
- Class names list (`classes`)

---

## 🚀 Deployment
The model was deployed using Streamlit, providing a visually engaging Marvel-themed web interface.

### Features:
- 🔴 Cinematic red-black theme inspired by Marvel Studios  
- 🖼️ Upload any Marvel hero image (JPG/PNG)  
- 🧠 Instant prediction with model confidence  
- 📊 Confidence bar chart highlighting prediction strength  
- ⚡ Lightweight and responsive design  

To launch locally:
```bash
streamlit run app.py
