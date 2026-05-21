import streamlit as st
import pickle
import pandas as pd
import os

# Set page configuration
st.set_page_config(
    page_title="Crop Recommendation System",
    page_icon="🌱",
    layout="centered"
)

# Load the trained model
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model.pkl: {e}")
    st.stop()

# Title and description
st.title("🌱 Smart Crop Recommendation System")
st.markdown("""
This application uses your trained Decision Tree Machine Learning model to recommend the best crop to cultivate 
based on specific soil characteristics and environmental conditions.
""")

st.subheader("Enter Soil & Environmental Conditions:")

# Create a clean 2-column layout for input fields
col1, col2 = st.columns(2)

with col1:
    n = st.number_input("Nitrogen (N) content in soil", min_value=0, max_value=200, value=50, step=1)
    p = st.number_input("Phosphorus (P) content in soil", min_value=0, max_value=200, value=50, step=1)
    k = st.number_input("Potassium (K) content in soil", min_value=0, max_value=200, value=50, step=1)
    ph = st.number_input("pH value of the soil", min_value=0.0, max_value=14.0, value=6.5, step=0.1)

with col2:
    temp = st.number_input("Temperature (in Celsius)", min_value=0.0, max_value=60.0, value=25.0, step=0.5)
    humidity = st.number_input("Relative Humidity (in %)", min_value=0.0, max_value=100.0, value=80.0, step=0.5)
    rainfall = st.number_input("Rainfall (in mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)

# High-quality online image URLs mapped to lowercase, clean keys
crop_images = {
    "apple": "https://images.unsplash.com/photo-1560806887-1e4cd0b6cbd6?w=600",
    "banana": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=600",
    "blackgram": "https://images.unsplash.com/photo-1585994437213-91dbfb3431ca?w=600",
    "chickpea": "https://images.unsplash.com/photo-1547058881-aa0edd92aab3?w=600",
    "coconut": "https://images.unsplash.com/photo-1560185007-c5ca9d2c014d?w=600",
    "coffee": "https://images.unsplash.com/photo-1514432324607-a09d9b4aefdd?w=600",
    "cotton": "https://images.unsplash.com/photo-1594488736340-ff882585f812?w=600",
    "grapes": "https://images.unsplash.com/photo-1537640538966-79f369143f8f?w=600",
    "jute": "https://images.unsplash.com/photo-1595855759920-86582396756a?w=600",
    "kidneybeans": "https://images.unsplash.com/photo-1606923829579-0ac9986a3e31?w=600",
    "lentil": "https://images.unsplash.com/photo-1515942400420-2b98fed1f515?w=600",
    "maize": "https://images.unsplash.com/photo-1551754655-cd27e38d2076?w=600",
    "mango": "https://images.unsplash.com/photo-1553279768-865429fa0078?w=600",
    "mothbeans": "https://images.unsplash.com/photo-1595855759920-86582396756a?w=600",
    "mungbean": "https://images.unsplash.com/photo-1595855759920-86582396756a?w=600",
    "muskmelon": "https://images.unsplash.com/photo-1571771894821-ce9b6c11b08e?w=600",
    "orange": "https://images.unsplash.com/photo-1547514701-42782101795e?w=600",
    "papaya": "https://images.unsplash.com/photo-1526318896980-cf78c088247c?w=600",
    "pigeonpeas": "https://images.unsplash.com/photo-1595855759920-86582396756a?w=600",
    "pomegranate": "https://images.unsplash.com/photo-1582979512210-99b6a53386f9?w=600",
    "rice": "https://images.unsplash.com/photo-1536257104079-aa99c6460a5a?w=600",
    "watermelon": "https://images.unsplash.com/photo-1587049352846-4a222e784d38?w=600"
}

# Recommendation Button
if st.button("Recommend Best Crop", type="primary"):
    # Format inputs into a DataFrame with exact feature names
    input_df = pd.DataFrame(
        [[n, p, k, temp, humidity, ph, rainfall]], 
        columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    )
    
    # Generate raw prediction
    raw_prediction = model.predict(input_df)[0]
    
    # CRITICAL FIX: Strip any hidden whitespaces and cast to lowercase to match keys perfectly
    prediction_cleaned = str(raw_prediction).strip().lower()
    
    st.markdown("---")
    st.success(f"### 🎉 Recommended Crop: **{prediction_cleaned.upper()}**")
    
    # Create an empty container to force fresh image rendering
    image_container = st.container()
    
    with image_container:
        local_img_path = os.path.join("images", f"{prediction_cleaned}.jpg")
        
        if os.path.exists(local_img_path):
            st.image(local_img_path, caption=f"Optimal crop choice: {prediction_cleaned.capitalize()}", use_container_width=True)
        elif prediction_cleaned in crop_images:
            # Append a small cache-busting flag to the URL so Streamlit updates the visual frame immediately
            img_url = f"{crop_images[prediction_cleaned]}&refresh=true"
            st.image(img_url, caption=f"Optimal crop choice: {prediction_cleaned.capitalize()}", use_container_width=True)
        else:
            st.info(f"No visualization image available in directory or mapping for '{prediction_cleaned}'.")
