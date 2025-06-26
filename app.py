import os
from dotenv import load_dotenv
load_dotenv()
import openai
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


gpt_api_key = os.getenv("gpt_api_key")

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

# Upload image
uploaded_file = st.file_uploader(
    "Upload a photo of your meal",
    type=["jpg", "jpeg", "png", "heic"]  # ← add "heic" here
)


# Load model & extractor (cached to avoid reloading every time)
@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    model = AutoModelForImageClassification.from_pretrained("eslamxm/vit-base-food101")
    return extractor, model

extractor, model = load_model()

# Identify food with GPT
def identify_food_with_gpt(image: Image.Image, openai_api_key: str):
    openai.api_key = openai_api_key

    import base64
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    response = openai.ChatCompletion.create(
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

# Get nutrition info with GPT
def get_nutrition_with_gpt(food_name: str, openai_api_key: str):
    openai.api_key = openai_api_key

    prompt = f"""
    You are a nutritionist advising a prediabetic patient.

    Please do the following for a typical serving of: "{food_name}":

    1. Provide **Calories**, **Protein (g)**, **Carbohydrates (g)**, and **Fat (g)** in bullet points.
    2. Estimate the **Glycemic Index (GI)** or **Glycemic Load (GL)**, if known.
    3. ✅ Give a **simple recommendation** for someone with prediabetes:
    - Is this food good, neutral, or risky?
    - Explain *why*
    - Suggest a healthier alternative if needed

    Keep it short and structured. Even if unsure about nutrition details, always include a recommendation for prediabetes.
    """


    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=300
    )

    return response.choices[0].message.content.strip()

# Display uploaded image and prediction
if uploaded_file:
    try:
        # Read the uploaded file into bytes
        file_bytes = uploaded_file.read()

        # Try to open the image
        if uploaded_file.name.lower().endswith(".heic"):
            heif_file = pillow_heif.read_heif(io.BytesIO(file_bytes))
            image = Image.frombytes(
                heif_file.mode, heif_file.size, heif_file.data, "raw"
            )
        else:
            image = Image.open(io.BytesIO(file_bytes))

        st.image(image, caption="Your uploaded meal", use_column_width=True)

        st.subheader("📈 Optional: Enter Your Glucose Level (1 Hour After Eating)")
        glucose_input = st.number_input(
            "Post-meal glucose (mg/dL)", 
            min_value=0, max_value=500, step=1, 
            help="Enter your blood glucose level 1 hour after eating"
)

        if st.button("Analyze and Log Food"):
            with st.spinner("Analyzing..."):
                try:                    
                    st.success(f"🍽️ GPT-4 thinks this is: **{food_name}**")

                    # Get nutrition info and advice
                    nutrition_info = get_nutrition_with_gpt(food_name, gpt_api_key)

                    # Default values
                    color = "gray"
                    label = "ℹ️ No GI info detected"

                    # Try to extract GI value
                    gi_match = re.search(r"(?i)glycemic index.*?(\d{2,3})", nutrition_info)
                    if gi_match:
                        gi_value = int(gi_match.group(1))

                        if gi_value < 55:
                            color = "green"
                            label = f"✅ Low GI ({gi_value}) – Good for prediabetes"
                        elif 55 <= gi_value <= 69:
                            color = "orange"
                            label = f"🟠 Medium GI ({gi_value}) – Use caution"
                        elif gi_value >= 70:
                            color = "red"
                            label = f"⚠️ High GI ({gi_value}) – Risky for prediabetes"

                    # Display result
                    st.markdown(f"<h4 style='color:{color};'>{label}</h4>", unsafe_allow_html=True)
                    st.write("🥗 GPT-estimated nutrition:")
                    st.markdown(nutrition_info)

                    def extract_nutrition(text):
                        def find(pattern):
                            match = re.search(pattern, text, re.IGNORECASE)
                            return match.group(1).strip() if match else "N/A"

                        return {
                            "calories": find(r"Calories:\s*~?(\d+)"),
                            "protein": find(r"Protein:\s*~?([\d\.]+)"),
                            "carbs": find(r"Carbohydrates:\s*~?([\d\.]+)"),
                            "fat": find(r"Fat:\s*~?([\d\.]+)"),
                            "gi": re.search(r"(?i)glycemic index.*?(\d{2,3})", text).group(1) if re.search(r"(?i)glycemic index.*?(\d{2,3})", text) else "N/A"
                        }
                    data = extract_nutrition(nutrition_info)
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
    import pandas as pd
    df = pd.read_csv(LOG_FILE)
    st.dataframe(df)
    st.download_button("Download Log as CSV", data=df.to_csv(index=False), file_name="food_log.csv")
else:
    st.info("No meals logged yet!")

