# Abkhaz POS Tagger

This project implements a **Part-of-Speech tagging system for the Abkhaz language** using XLM-RoBERTa. It includes data preprocessing, training, evaluation, and a live feedback collection and retraining loop via a **Streamlit web app**. It is designed to work with **low-resource, morphologically rich languages**, with a focus on **continuous learning from user feedback**.

---

## 🧑 Author
**Ümit Altar Binici**

---

## 📁 Directory Structure
```plaintext
|  
├── main.py               # Full training & evaluation pipeline
├── preprocessor.py       # Prepares dataset: tokenization + label alignment
├── dataloader.py         # Loads preprocessed data into PyTorch DataLoaders
├── model.py              # POS tagger model (XLM-RoBERTa + classifier)
├── trainer.py            # Training logic with validation + LR scheduling
├── evaluator.py          # Evaluation on the test set
├── frontend.py           # Streamlit app with live tagging, feedback, retraining
├── tokenized_abkhaz_dataset.pth   # Preprocessed dataset
├── feedback_log.txt      # Collected feedback (uploaded to Hugging Face)
├── pos_model.pth  # Model checkpoint (downloaded from Hugging Face)
├── README.md             # This file
```

---

## 🧪 Versions & Dependencies
- **Python:** 3.11.5
- **Libraries:**
  - `torch` 2.0.1
  - `transformers` 4.34.0
  - `datasets` 2.17.1
  - `streamlit`, `pandas`, `matplotlib`
  - `huggingface_hub` for remote model/file syncing

---

## 🧠 Features
### ✅ Transformer-Based POS Tagging
- Uses `xlm-roberta-base` as backbone
- Frozen lower layers, fine-tuned classifier

### ✅ Feedback-Aware Retraining
- Users can correct tags in the UI
- Feedback is saved + uploaded to Hugging Face Hub
- App supports retraining based on cumulative feedback

### ✅ Continuous Deployment (Streamlit Cloud)
- Pulls latest model from Hugging Face Hub
- Uploads new versions via API token securely

---

## 🚀 How to Run
### 1. Preprocess Abkhaz Dataset
```bash
python preprocessor.py
```
Outputs: `tokenized_abkhaz_dataset.pth`

### 2. Train + Evaluate Model (Full Pipeline)
```bash
python main.py
```
- Ensures dataset is preprocessed
- Trains model and evaluates it on test set
- Outputs: `pos_model.pth`, evaluation logs

### 3. Launch Streamlit Demo App (for tagging & feedback)
```bash
streamlit run frontend.py
```
Ensure `.streamlit/secrets.toml` is set up with:
```toml
[huggingface]
token = "hf_xxxx"
```

The app allows real-time tagging and lets users correct errors. Feedback is stored remotely and used for fine-tuning.

---

## 📬 Feedback Integration
- Every submission logs corrected POS tags.
- Once threshold (e.g. 10 entries) is reached, user can trigger retraining.
- Updated model is pushed back to Hugging Face and pulled live by the app.

---

## 📚 Learning Outcomes
- Fine-tuning multilingual Transformers for morphologically rich languages
- End-to-end data processing, training, and evaluation with PyTorch
- Interactive NLP system with user-in-the-loop learning
- Deployment pipeline from model training to continuous feedback loop

---

## 🌍 Acknowledgements
- [Universal Dependencies: Abkhaz Treebank](https://universaldependencies.org/treebanks/ab_abnc/index.html)
- Hugging Face, Streamlit, PyTorch, Transformers community

