import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from transformers import XLMRobertaTokenizerFast
from model import POSModel
from preprocessor import id2tag
from trainer import train_model
from dataloader import load_data
from evaluator import evaluate_model
from huggingface_hub import hf_hub_download, upload_file, login
import os

# Authenticate to Hugging Face using Streamlit secrets
hf_token = st.secrets["huggingface"]["token"]
login(token=hf_token)

# Hugging Face Hub Config
HF_REPO_ID = "altarbinici/Abkhaz-POS-Tagger"
HF_FEEDBACK_REPO = "altarbinici/Abkhaz-POS-Feedback"
MODEL_FILENAME = "pos_model_latest.pth"
FEEDBACK_FILENAME = "feedback_log.txt"

# Load tokenizer and model
num_tags = len(id2tag)
tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
model_path = hf_hub_download(repo_id=HF_REPO_ID, filename=MODEL_FILENAME)
model = POSModel(num_tags)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
model.eval()

# Annotate user input
def annotate_text(input_text):
    tokens = input_text.strip().split()
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=128
    )

    with torch.no_grad():
        logits = model(input_ids=encoded["input_ids"], attention_mask=encoded["attention_mask"])
        predictions = torch.argmax(logits, dim=-1).squeeze().tolist()

    word_ids = encoded.word_ids()
    output = []
    prev_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is not None and word_idx != prev_word_idx:
            tag_id = predictions[idx]
            tag = id2tag.get(tag_id, "UNK")
            output.append((tokens[word_idx], tag))
            prev_word_idx = word_idx

    return output

# Streamlit Interface
st.set_page_config(page_title="Abkhaz POS Tagger", layout="centered")
st.title("📝 Abkhaz POS Tagging Demo")
st.write("Enter an Abkhaz sentence below to see predicted part-of-speech tags.")

user_input = st.text_input("Enter Abkhaz sentence:", "")

if user_input:
    annotations = annotate_text(user_input)
    st.markdown("### 🔍 Predicted POS Tags:")
    for token, tag in annotations:
        st.markdown(f"**{token}** — *{tag}*")

    st.markdown("### 🧠 Feedback")
    feedback = st.radio("Were the POS tags correct?", ["Yes", "No"])
    corrected_tags = []

    if feedback == "No":
        st.markdown("#### ✏️ Provide corrections for incorrect tags:")
        for idx, (token, predicted_tag) in enumerate(annotations):
            col1, col2 = st.columns([2, 2])
            with col1:
                st.markdown(f"**{token}** ({predicted_tag})")
            with col2:
                corrected = st.text_input(f"Correct tag for '{token}'", key=f"correction_{idx}")
                corrected_tags.append(corrected if corrected else predicted_tag)
    else:
        corrected_tags = [tag for _, tag in annotations]

    if st.button("Submit Feedback"):
        with open(FEEDBACK_FILENAME, "a", encoding="utf-8") as f:
            f.write(f"Input: {user_input}\nPredicted: {annotations}\nFeedback: {feedback}\nCorrection: {corrected_tags}\n---\n")
        upload_file(
            path_or_fileobj=FEEDBACK_FILENAME,
            path_in_repo=FEEDBACK_FILENAME,
            repo_id=HF_FEEDBACK_REPO,
            repo_type="dataset"
        )
        st.success("✅ Thank you for your feedback! It has been saved and uploaded.")

# Evaluation and feedback summary
try:
    feedback_log_path = hf_hub_download(repo_id=HF_FEEDBACK_REPO, filename=FEEDBACK_FILENAME, repo_type="dataset")
    with open(feedback_log_path, "r", encoding="utf-8") as f:
        feedback_entries = f.read().split("---\n")
        total_feedback = len([entry for entry in feedback_entries if "Input:" in entry])
        st.markdown(f"### 📊 Feedback Summary ({total_feedback} entries)")

        corrections = []
        for entry in feedback_entries:
            if "Correction:" in entry:
                try:
                    tags_line = entry.split("Correction:")[1].strip().split("\n")[0]
                    tags = eval(tags_line) if tags_line.startswith("[") else tags_line.split()
                    corrections.extend(tags)
                except Exception:
                    continue

        if corrections:
            correction_counts = pd.Series(corrections).value_counts()
            st.bar_chart(correction_counts)

        if total_feedback >= 10:
            st.warning("⚠️ Model has received over 10 feedback entries. Consider retraining.")
            if st.button("🔁 Retrain Model with Feedback"):
                with st.spinner("Retraining with feedback and UD data blend..."):
                    train_loader, val_loader, test_loader, num_tags = load_data()
                    model_path = train_model(train_loader, val_loader, num_tags)
                    upload_file(
                        path_or_fileobj=model_path,
                        path_in_repo=MODEL_FILENAME,
                        repo_id=HF_REPO_ID
                    )
                st.success("✅ Model retrained and updated on Hugging Face.")
                st.rerun()

        if st.button("🧪 Evaluate Current Model"):
            with st.spinner("Running evaluation on test set..."):
                test_loader = load_data()[2]
                evaluate_model(test_loader, num_tags, model_path)
except Exception as e:
    st.error(f"⚠️ Could not load feedback summary: {e}")
