# from fastapi import FastAPI, UploadFile, File, Form
# from fastapi.middleware.cors import CORSMiddleware
# import numpy as np
# import joblib
# from PIL import Image
# from io import BytesIO
# import re
# import torch
# import clip

# # -----------------------------
# # INIT
# # -----------------------------
# app = FastAPI()
# SAVE_DIR = "artifacts"

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using device: {DEVICE}")

# # -----------------------------
# # CORS (VERY IMPORTANT)
# # -----------------------------
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],   # allow frontend
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # -----------------------------
# # LOAD MODELS
# # -----------------------------
# print("Loading ML models...")

# text_pca = joblib.load(f"{SAVE_DIR}/text_pca.pkl")
# img_pca  = joblib.load(f"{SAVE_DIR}/img_pca.pkl")

# cat_model = joblib.load(f"{SAVE_DIR}/catboost.pkl")
# lgb_model = joblib.load(f"{SAVE_DIR}/lgb.pkl")

# tfidf = joblib.load(f"{SAVE_DIR}/tfidf.pkl")
# kmeans = joblib.load(f"{SAVE_DIR}/kmeans.pkl")

# print("✅ ML models loaded")

# # -----------------------------
# # LOAD CLIP
# # -----------------------------
# print("Loading CLIP...")

# clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
# clip_model.eval()

# print("✅ CLIP loaded")

# # -----------------------------
# # TEXT MODEL
# # -----------------------------
# from sentence_transformers import SentenceTransformer
# text_model = SentenceTransformer('all-MiniLM-L6-v2')

# # -----------------------------
# # TEXT PROCESSING
# # -----------------------------
# def process_text(text: str):
#     text = text.lower()
#     text = re.sub(r'bullet point \d+:', 'bullet point:', text)
#     text = re.sub(r'\s+', ' ', text).strip()
#     return text

# # -----------------------------
# # TABULAR FEATURES
# # -----------------------------
# def extract_tab_features(text):

#     text_len = len(text.split())

#     unit = 1 if any(u in text for u in ["ounce", "oz", "ml", "kg", "gram"]) else 0

#     pack = 1
#     match = re.search(r'pack of (\d+)', text)
#     if match:
#         pack = int(match.group(1))

#     brand = 0

#     return np.array([[brand, unit, pack, text_len]])

# # -----------------------------
# # EMBEDDINGS
# # -----------------------------
# def get_text_embedding(text):
#     return text_model.encode([text])[0]

# def get_image_embedding(image):
#     img = preprocess(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         emb = clip_model.encode_image(img)

#     return emb.cpu().numpy().flatten()

# # -----------------------------
# # FEATURE BUILDER
# # -----------------------------
# def build_features(text, image):

#     text = process_text(text)

#     text_emb = get_text_embedding(text)
#     img_emb  = get_image_embedding(image)

#     text_emb_pca = text_pca.transform([text_emb])
#     img_emb_pca  = img_pca.transform([img_emb])

#     tab = extract_tab_features(text)
#     cat = kmeans.predict(tfidf.transform([text])).reshape(-1,1)

#     X = np.hstack([text_emb_pca, img_emb_pca, tab, cat])

#     # DEBUG (optional but useful)
#     print("\n====== DEBUG ======")
#     print("TEXT:", text[:100])
#     print("TEXT EMB (first 5):", text_emb[:5])
#     print("IMG EMB (first 5):", img_emb[:5])
#     print("TEXT PCA (first 5):", text_emb_pca[0][:5])
#     print("IMG PCA (first 5):", img_emb_pca[0][:5])
#     print("CAT:", cat)
#     print("TAB:", tab)
#     print("===================\n")

#     return X

# # -----------------------------
# # API
# # -----------------------------
# @app.post("/predict")
# async def predict(
#     text: str = Form(...),
#     file: UploadFile = File(...)
# ):
#     try:
#         contents = await file.read()

#         if not contents:
#             return {"error": "Empty image file"}

#         image = Image.open(BytesIO(contents)).convert("RGB")

#         X = build_features(text, image)

#         pred_cat = cat_model.predict(X)
#         pred_lgb = lgb_model.predict(X)

#         final_log = pred_cat
#         price = np.expm1(final_log[0])

#         return {
#             "predicted_price": float(price)
#         }

#     except Exception as e:
#         print("ERROR:", str(e))
#         return {"error": str(e)}

# # -----------------------------
# # ROOT
# # -----------------------------
# @app.get("/")
# def home():
#     return {"message": "API running 🚀"}


from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
from PIL import Image
from io import BytesIO
import re
import torch
import clip

# -----------------------------
# INIT
# -----------------------------
app = FastAPI()
SAVE_DIR = "artifacts"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# LOAD MODELS
# -----------------------------
print("Loading ML models...")

text_pca = joblib.load(f"{SAVE_DIR}/text_pca.pkl")
img_pca  = joblib.load(f"{SAVE_DIR}/img_pca.pkl")

cat_model = joblib.load(f"{SAVE_DIR}/catboost.pkl")
lgb_model = joblib.load(f"{SAVE_DIR}/lgb.pkl")

tfidf = joblib.load(f"{SAVE_DIR}/tfidf.pkl")
kmeans = joblib.load(f"{SAVE_DIR}/kmeans.pkl")

print("✅ ML models loaded")

# -----------------------------
# LOAD CLIP
# -----------------------------
print("Loading CLIP...")

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()

print("✅ CLIP loaded")

# -----------------------------
# TEXT MODEL
# -----------------------------
from sentence_transformers import SentenceTransformer

text_model = SentenceTransformer('all-MiniLM-L6-v2')

# -----------------------------
# TEXT PROCESSING
# -----------------------------
def process_text(text: str):
    text = text.lower()
    text = re.sub(r'bullet point \d+:', 'bullet point:', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# -----------------------------
# TABULAR FEATURES
# -----------------------------
def extract_tab_features(text):

    text_len = len(text.split())

    unit = 1 if any(
        u in text for u in ["ounce", "oz", "ml", "kg", "gram"]
    ) else 0

    pack = 1

    match = re.search(r'pack of (\d+)', text)

    if match:
        pack = int(match.group(1))

    brand = 0

    return np.array([[brand, unit, pack, text_len]])

# -----------------------------
# EMBEDDINGS
# -----------------------------
def get_text_embedding(text):
    return text_model.encode([text])[0]

def get_image_embedding(image):

    img = preprocess(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        emb = clip_model.encode_image(img)

    return emb.cpu().numpy().flatten()

# -----------------------------
# FEATURE BUILDER
# -----------------------------
def build_features(text, image):

    text = process_text(text)

    text_emb = get_text_embedding(text)
    img_emb  = get_image_embedding(image)

    text_emb_pca = text_pca.transform([text_emb])
    img_emb_pca  = img_pca.transform([img_emb])

    tab = extract_tab_features(text)

    cat = kmeans.predict(
        tfidf.transform([text])
    ).reshape(-1,1)

    X = np.hstack([
        text_emb_pca,
        img_emb_pca,
        tab,
        cat
    ])

    # DEBUG
    print("\n====== DEBUG ======")
    print("TEXT:", text[:100])
    print("TEXT EMB (first 5):", text_emb[:5])
    print("IMG EMB (first 5):", img_emb[:5])
    print("TEXT PCA (first 5):", text_emb_pca[0][:5])
    print("IMG PCA (first 5):", img_emb_pca[0][:5])
    print("CAT:", cat)
    print("TAB:", tab)
    print("===================\n")

    return X

# -----------------------------
# API
# -----------------------------
@app.post("/predict")
async def predict(
    text: str = Form(...),
    file: UploadFile = File(...)
):
    try:

        contents = await file.read()

        if not contents:
            return {"error": "Empty image file"}

        image = Image.open(
            BytesIO(contents)
        ).convert("RGB")

        X = build_features(text, image)

        pred_cat = cat_model.predict(X)
        pred_lgb = lgb_model.predict(X)

        # --------------------------------
        # FINAL PREDICTION
        # --------------------------------

        # optional ensemble:
        # final_log = 0.5 * pred_cat + 0.5 * pred_lgb

        final_log = pred_cat

        price = np.expm1(final_log[0])

        # --------------------------------
        # RANGE USING LOG RMSE
        # --------------------------------

        LOG_RMSE = 0.75

        lower = price * np.exp(-LOG_RMSE)
        upper = price * np.exp(LOG_RMSE)

        return {
            "predicted_price": round(float(price), 2),

            "predicted_range": {
                "min": round(float(lower), 2),
                "max": round(float(upper), 2)
            }
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}

# -----------------------------
# ROOT
# -----------------------------
@app.get("/")
def home():
    return {"message": "API running 🚀"}

import uvicorn

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)