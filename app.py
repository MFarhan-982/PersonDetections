import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import os

# ✅ Correct path to your model
model_path = r"C:\Users\DELL\PycharmProjects\CORVIT\best (1).pt"

# ✅ Check if model exists
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

# ✅ Load YOLO model
model = YOLO(model_path)

# ✅ Streamlit page settings
st.set_page_config(page_title="Persons Detection", layout="centered")

# Sidebar
st.sidebar.title("🔖 About")
st.sidebar.markdown("""
**MUHAMMAD FARHAN RANA**  
📧 mfarhanrana982@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/muhammadfarhanrana)
""")

# Title
st.title("👤 Persons Detection Model")
st.markdown("Upload an image to detect **persons** using a YOLOv8 model.")

# File uploader
uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        image.save(tmp_file.name)
        image_path = tmp_file.name

    # Run detection
    with st.spinner("🔍 Detecting..."):
        results = model(image_path)
        boxes_img = results[0].plot()

    # Show result image
    st.image(boxes_img, caption="Detection Results", use_column_width=True)

    # Show detected classes
    st.subheader("📋 Detected Objects")
    for box in results[0].boxes.data.tolist():
        class_id = int(box[5])
        class_name = results[0].names[class_id]
        conf = float(box[4])
        st.write(f"🟢 {class_name}: {conf:.2f}")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using [Ultralytics YOLOv8](https://docs.ultralytics.com/) and Streamlit.")
