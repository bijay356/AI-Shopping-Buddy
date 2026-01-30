import streamlit as st
import torch
import clip
import numpy as np
from PIL import Image
import os

# ---------------- App Config ----------------
st.set_page_config(
    page_title="AI Shopping Buddy",
    layout="centered"
)

st.title("🛍️ AI Shopping Buddy")
st.write("Upload a clothing image and get similar fashion recommendations using AI.")

# ---------------- Load CLIP Model ----------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device

model, preprocess, device = load_model()

# ---------------- Dataset Folder ----------------
DATA_DIR = "shirts"   # folder containing dataset images

if not os.path.exists(DATA_DIR):
    st.error("❌ 'shirts' folder not found. Please upload images to a 'shirts' folder in GitHub.")
    st.stop()

# ---------------- Upload Image ----------------
uploaded_file = st.file_uploader(
    "Upload a shirt image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    # Show uploaded image
    query_image = Image.open(uploaded_file).convert("RGB")
    st.image(query_image, caption="Query Image", width=300)

    # Encode query image
    image_input = preprocess(query_image).unsqueeze(0).to(device)
    with torch.no_grad():
        query_embedding = model.encode_image(image_input)
        query_embedding /= query_embedding.norm(dim=-1, keepdim=True)

    # ---------------- Load Dataset Images ----------------
    image_embeddings = []
    image_names = []

    for img_name in os.listdir(DATA_DIR):
        img_path = os.path.join(DATA_DIR, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
            img_input = preprocess(img).unsqueeze(0).to(device)

            with torch.no_grad():
                emb = model.encode_image(img_input)
                emb /= emb.norm(dim=-1, keepdim=True)

            image_embeddings.append(emb.cpu().numpy())
            image_names.append(img_name)

        except:
            continue

    if len(image_embeddings) == 0:
        st.error("No valid images found in shirts folder.")
        st.stop()

    image_embeddings = np.vstack(image_embeddings)

    # ---------------- Similarity ----------------
    similarities = (query_embedding.cpu().numpy() @ image_embeddings.T)[0]
    top_indices = similarities.argsort()[-5:][::-1]

    # ---------------- Results ----------------
    st.subheader("🔍 Similar Recommendations")

    cols = st.columns(len(top_indices))
    for col, idx in zip(cols, top_indices):
        rec_img = Image.open(os.path.join(DATA_DIR, image_names[idx]))
        col.image(rec_img, caption=f"Score: {similarities[idx]:.2f}", use_column_width=True)

    # ---------------- Style Advice ----------------
    style_texts = [
        "Minimal casual everyday wear",
        "Smart casual outfit for office",
        "Streetwear fashion with layered clothing",
        "Formal outfit for meetings or events",
        "Relaxed home or lounge wear"
    ]

    text_tokens = clip.tokenize(style_texts).to(device)
    with torch.no_grad():
        text_embeddings = model.encode_text(text_tokens)
        text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    style_scores = (query_embedding @ text_embeddings.T).squeeze(0)
    best_style_idx = style_scores.argmax().item()

    style_advice = {
        "Minimal casual everyday wear":
            "Perfect for daily casual looks. Pair it with jeans and sneakers.",
        "Smart casual outfit for office":
            "Looks office-ready. Try chinos and loafers.",
        "Streetwear fashion with layered clothing":
            "Great for streetwear vibes. Layer with a hoodie or jacket.",
        "Formal outfit for meetings or events":
            "Style with tailored pants and formal shoes.",
        "Relaxed home or lounge wear":
            "Ideal for comfort. Pair with joggers or shorts."
    }

    st.subheader("💡 Style Advice")
    st.success(style_advice[style_texts[best_style_idx]])
