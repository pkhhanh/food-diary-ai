from dotenv import load_dotenv
load_dotenv(override=True)
import os
from openai import OpenAI
import re
import requests
import io
import pillow_heif
import streamlit as st
from PIL import Image
import torch
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
import csv
from datetime import datetime
import pandas as pd

# Get OpenAI key from environment
gpt_api_key = os.getenv("gpt_api_key")
client = OpenAI(api_key=gpt_api_key)

LOG_FILE = "food_log.csv"

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Timestamp", "Image Filename", "Food",
            "Calories", "Protein (g)", "Carbs (g)", "Fat (g)",
            "GI", "Risk Level", "Glucose (mg/dL)"
        ])

st.title("🍽️ AI Food Detector")

uploaded_file = st.file_uploader(
    "Upload a photo of your meal",
    type=["jpg", "jpeg", "png", "heic"]
)

@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("eslamxm/vit-base-food101")
    return extractor, model

extractor, model = load_model()

def identify_food_with_gpt(image: Image.Image):
    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What food is shown in this image? Respond with only the dish name, e.g., 'banana', 'chicken rice'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ],
        max_tokens=20
    )
    return response.choices[0].message.content.strip()

def get_nutrition_with_gpt(food_name: str):
    prompt = f"""
For a typical serving of "{food_name}", reply only in the following format:

Calories: <number> kcal
Protein: <number> g
Carbohydrates: <number> g
Fat: <number> g
Glycemic Index: <number or N/A>
Recommendation: <good/neutral/risky>. <one-sentence advice for prediabetic>
"""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=200,
    )
    return response.choices[0].message.content.strip()

def extract_nutrition(text):
    nutrition = {"calories": "N/A", "protein": "N/A", "carbs": "N/A", "fat": "N/A", "gi": "N/A", "recommendation": "N/A"}
    cal = re.search(r"Calories:\s*([0-9]+)", text, re.IGNORECASE)
    protein = re.search(r"Protein:\s*([0-9.]+)", text, re.IGNORECASE)
    carbs = re.search(r"Carbohydrates:\s*([0-9.]+)", text, re.IGNORECASE)
    fat = re.search(r"Fat:\s*([0-9.]+)", text, re.IGNORECASE)
    gi = re.search(r"Glycemic Index:\s*([0-9]+|N/A)", text, re.IGNORECASE)
    rec = re.search(r"Recommendation:\s*([^\n]+)", text, re.IGNORECASE)

    if cal: nutrition["calories"] = cal.group(1)
    if protein: nutrition["protein"] = protein.group(1)
    if carbs: nutrition["carbs"] = carbs.group(1)
    if fat: nutrition["fat"] = fat.group(1)
    if gi: nutrition["gi"] = gi.group(1)
    if rec: nutrition["recommendation"] = rec.group(1).strip()
    return nutrition

if uploaded_file:
    try:
        file_bytes = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".heic"):
            heif_file = pillow_heif.read_heif(io.BytesIO(file_bytes))
            image = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            )
        else:
            image = Image.open(io.BytesIO(file_bytes))

        st.image(image, caption="Your uploaded meal", use_container_width=True)

        st.subheader("📈 Optional: Enter Your Glucose Level (1 Hour After Eating)")
        glucose_input = st.number_input(
            "Post-meal glucose (mg/dL)",
            min_value=0, max_value=500, step=1,
            help="Enter your blood glucose level 1 hour after eating"
        )

        if st.button("Analyze and Log Food"):
            with st.spinner("Analyzing..."):
                try:
                    food_name = identify_food_with_gpt(image)
                    st.success(f"🍽️ GPT-4 thinks this is: **{food_name}**")

                    nutrition_info = get_nutrition_with_gpt(food_name)
                    data = extract_nutrition(nutrition_info)

                    # Set color and label for GI
                    color = "gray"
                    label = "ℹ️ No GI info detected"
                    gi_value = data["gi"]
                    try:
                        if gi_value != "N/A":
                            gi_value = int(gi_value)
                            if gi_value < 55:
                                color = "green"
                                label = f"✅ Low GI ({gi_value}) – Good for prediabetes"
                            elif 55 <= gi_value <= 69:
                                color = "orange"
                                label = f"🟠 Medium GI ({gi_value}) – Use caution"
                            else:
                                color = "red"
                                label = f"⚠️ High GI ({gi_value}) – Risky for prediabetes"
                    except:
                        pass

                    st.markdown(f"<h4 style='color:{color};'>{label}</h4>", unsafe_allow_html=True)
                    st.write("🥗 GPT-estimated nutrition:")
                    st.markdown(nutrition_info)

                    with open(LOG_FILE, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            uploaded_file.name,
                            food_name,
                            data["calories"],
                            data["protein"],
                            data["carbs"],
                            data["fat"],
                            data["gi"],
                            label,
                            glucose_input if glucose_input > 0 else "N/A"
                        ])
                except Exception as e:
                    st.error(f"Error: {e}")

    except Exception as e:
        st.error(f"Failed to process image: {e}")

st.header("📊 Your Food Diary")

if os.path.exists(LOG_FILE):
    df = pd.read_csv(LOG_FILE)
    st.dataframe(df)
    st.download_button("Download Log as CSV", data=df.to_csv(index=False), file_name="food_log.csv")
else:
    st.info("No meals logged yet!")
