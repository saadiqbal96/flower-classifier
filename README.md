# 🌸 Flower Image Classifier (PyTorch)

This project trains a deep learning model to classify flower images into one of 102 categories using transfer learning in PyTorch.

---

## 🧠 Project Overview

The model uses a **pretrained CNN (VGG16 or DenseNet121)** from `torchvision.models`, then replaces the classifier to train on the flower dataset.

Key Features:
- Transfer learning with pretrained network
- Data augmentation and normalization
- GPU-compatible training
- Model checkpoint saving & loading
- Prediction of top-K classes with probabilities

---

## 📁 Files in This Repository

| File | Description |
|------|--------------|
| `train.py` | Trains a model on a dataset and saves it as a checkpoint |
| `predict.py` | Loads a checkpoint and predicts an image class |
| `cat_to_name.json` | Maps flower category numbers to actual flower names |
| `requirements.txt` | Lists Python packages required to run the project |
| `.gitignore` | Files and folders GitHub should ignore (like large datasets) |
| `README.md` | Project explanation (this file) |
| `PyTorch.HTML.zip` | Part 1 development notebook exported as HTML (zipped) |

---

## ⚙️ How to Use

### 🧩 Train a Model

python train.py data_dir –arch vgg16 –learning_rate 0.001 –hidden_units 512 –epochs 5 –gpu

### 🔮 Predict a Flower

python predict.py path/to/image checkpoint.pth –top_k 5 –category_names cat_to_name.json –gpu
---

## 🧾 Rubric Checklist

**Part 1 – Development Notebook**
- ✅ All necessary packages imported  
- ✅ Data loaded, normalized, and augmented  
- ✅ Pretrained model loaded (VGG16 or DenseNet121)  
- ✅ Classifier trained and validated  
- ✅ Model tested and accuracy printed  
- ✅ Checkpoint saved and loaded successfully  
- ✅ Completed Jupyter Notebook exported as HTML (see `PyTorch.HTML.zip`)

**Part 2 – Command Line Application**
- ✅ `train.py` trains model, saves checkpoint  
- ✅ `predict.py` loads checkpoint and predicts class  
- ✅ Both scripts accept command-line arguments  
- ✅ Supports GPU training and inference  
- ✅ Supports JSON category name mapping  

---

## 🧰 Requirements

Install dependencies:

pip install -r requirements.txt

---

## 🪪 License

This project was created for educational purposes and may be used or modified freely for learning and teaching.
