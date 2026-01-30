import streamlit as st
import torch
import clip
import numpy as np
from PIL import Image
import os

# ---------- App UI ----------
st.set_page_config(page_title="AI Shopping Buddy", layout="centered")
st.title("🛍️ AI Shopping Buddy")
st.write("Upload a clothing image to get similar recommendations and style advice.")

# ---------- Load CLIP ----------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# ---------- Upload Image ----------
uploaded_file = st.file_uploader("Upload a shirt image", type=["jpg", "png", "jpeg"])

# ---------- Sample Dataset Folder ----------
DATA_DIR = "shirts"  # later you can change

if uploaded_file is not None:
    query_image = Image.open(uploaded_file)
    st.image(query_image, caption="Query Image", width=300)

    # Encode query image
    image_input = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(image_input)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    # Load dataset images
    image_embeddings = []
    image_names = []

    for img_name in os.listdir(DATA_DIR):
        img_path = os.path.join(DATA_DIR, img_name)
        img = preprocess(Image.open(img_path)).unsqueeze(0).to(device)

        with torch.no_grad():
            emb = model.encode_image(img)
            emb /= emb.norm(dim=-1, keepdim=True)

        image_embeddings.append(emb.cpu().numpy())
        image_names.append(img_name)

    image_embeddings = np.vstack(image_embeddings)

    # Similarity
    similarities = (query_embedding.cpu().numpy() @ image_embeddings.T)[0]
    top_indices = similarities.argsort()[-5:][::-1]

    st.subheader("🔍 Recommended Similar Items")
    for idx in top_indices:
        st.image(os.path.join(DATA_DIR, image_names[idx]), width=150)

    # Style advice (simple GenAI logic)
    st.subheader("👗 Style Advice")
    st.success("Streetwear vibe detected. Try layering with a hoodie or jacket.")
