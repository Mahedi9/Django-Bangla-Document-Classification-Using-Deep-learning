# 📄 Django-Based Bangla Document Classification Using Deep Learning

**Enhancing Bangla Document Classification Using a Hybrid Ensemble of Bangla-BERT and Bi-LSTM Models**  
**(IDAA 2025 – Published Research)**

This repository contains a **production-ready Django web application** for Bangla document classification using a **hybrid ensemble of Bangla-BERT and Bi-LSTM models**.  
The system is designed for **research, academic demonstration, and real-world deployment** of Bangla NLP models.

---

## 🚀 How to Run the Project (Quick Start)

### 1️⃣ Clone the Repository

    git clone https://github.com/Mahedi9/Django-Bangla-Document-Classification-Using-Deep-learning.git
    cd Django-Bangla-Document-Classification-Using-Deep-learning

### 2️⃣ Install Dependencies

    pip install -r requirements.txt

### 3️⃣ Run Database Migrations

    python manage.py migrate

### 4️⃣ Start the Django Server

    python manage.py runserver

### 5️⃣ Open in Browser

    http://127.0.0.1:8000/

---
<img width="1280" height="633" alt="image" src="https://github.com/user-attachments/assets/352724fd-d0d2-4987-8e9d-df180553f7ae" />
<img width="1280" height="680" alt="image" src="https://github.com/user-attachments/assets/2566cf06-a79f-4fda-8463-ff087a12a87a" />
<img width="1280" height="663" alt="image" src="https://github.com/user-attachments/assets/464ec4f4-34f2-4342-960f-176ec94780fe" />



## 🧠 Project Overview

The application classifies Bangla news articles into **eight categories** using a **hybrid ensemble learning strategy**:

- **Bi-LSTM** for sequential text representation
- **Bangla-BERT** for contextual semantic understanding
- **weighted voting ensemble** with confidence-based decision logic

The system produces:
- Final prediction with confidence
- Model-wise predictions
- Probability distribution visualizations
- Ambiguity warnings when predictions are uncertain

---

## ✨ Core Features

- Hybrid ensemble of **Bi-LSTM + Bangla-BERT**
- Confidence threshold–based ambiguity detection
- Model-wise prediction cards with probabilities
- Probability bar charts for each model
- Class-wise probability table
- Minimum input validation (≥ 20 Bangla words)
- Clean, responsive UI using Bootstrap
- Modular Django ML architecture

---

## 🧠 Supported News Categories

- Economy  
- Education  
- Entertainment  
- International  
- National  
- Politics  
- Science_Technology  
- Sports  

---

## 🧩 System Architecture

bangla_classifier_django/  
├── bangla_classifier_django/  
│   ├── settings.py  
│   ├── urls.py  
│   ├── asgi.py  
│   └── wsgi.py  
│  
├── classifier/  
│   ├── ml/  
│   │   ├── loader.py          (Model loading)  
│   │   ├── preprocess.py     (Text cleaning & stopword removal)  
│   │   └── predict.py        (Inference & ensemble logic)  
│   │  
│   ├── templates/classifier/  
│   │   └── index.html        (UI template)  
│   │  
│   ├── views.py              (Controller logic)  
│   ├── urls.py  
│   └── apps.py  
│  
├── models/  
│   ├── bilstm_model.keras  
│   ├── bilstm_tokenizer.pkl  
│   ├── bilstm_label_encoder.pkl  
│   └── bangla_bert_model/  
│  
├── static/  
├── manage.py  
├── requirements.txt  
├── .gitignore  
└── .gitattributes  

---

## 🔍 Prediction Workflow

1. User submits a Bangla news article (minimum 20 words)
2. Text is preprocessed (cleaning, stemming, stopword removal)
3. Bi-LSTM and Bangla-BERT generate class probabilities
4. Probabilities are combined using soft voting
5. Confidence threshold determines final decision
6. Results are visualized and displayed

---

## 📊 Experimental Results (From Published Paper)

- **Dataset**: Potrika Bangla News Dataset  
- **Total Articles**: 329,110  
- **Number of Classes**: 8  
- **Hybrid Ensemble Accuracy**: **97.16%**

---

## 📄 Research Publication

**Title**  
Enhancing Bangla Document Classification Using a Hybrid Ensemble of Bangla-BERT and Bi-LSTM Models  

**Conference**  
International Conference on Intelligent Data Analysis and Applications (IDAA 2025)

**Venue**  
Daffodil International University, Dhaka, Bangladesh

---

## 👤 Author

**Mahedi Hasan Emon**  
Researcher | Bangla NLP | Deep Learning  
Published Author – IDAA 2025  

---

## 🙏 Acknowledgements

- Potrika Bangla News Dataset  
- Hugging Face Transformers  
- TensorFlow  
- PyTorch  
- bnltk  
- Daffodil International University  

---

## 📌 License

This project is released for **research and educational use**.  
For commercial usage, please contact the author.

---
