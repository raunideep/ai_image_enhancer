import streamlit as st
from PIL import Image, ImageEnhance
import numpy as np
import cv2
from io import BytesIO

st.set_page_config(page_title="AI Image Upscaler", layout="centered")

st.title("🚀 AI Image Upscaler & Enhancer")
st.write("Upload a low-resolution or blurry image and drag the slider to enhance it dynamically.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    orig_width, orig_height = image.size
    orig_size_kb = uploaded_file.size / 1024
    orig_size_str = f"{orig_size_kb:.2f} KB" if orig_size_kb < 1024 else f"{orig_size_kb/1024:.2f} MB"

    st.markdown("---")
    # Slider moved to main screen right above the images
    st.subheader("⚙️ Dynamic Controls")
    enhancement_power = st.slider(
        "⬅️ Less Enhance | More Enhance ➡️",
        min_value=1.0,
        max_value=3.0,
        value=1.5,
        step=0.1,
        help="Left side mein kam enhancement aur right side mein maximum enhancement hogi."
    )
    st.markdown("---")

    # Scale factor based on slider position
    scale_factor = 2 if enhancement_power < 2.0 else 4

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        st.caption(f"📏 Resolution: {orig_width} x {orig_height} px\n💾 Size: {orig_size_str}")

    # Real-time processing based on slider movement
    with st.spinner("Processing image..."):
        # Convert PIL to OpenCV format
        img_cv = np.array(image)
        if img_cv.ndim == 2:  # Grayscale
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_GRAY2BGR)
        elif img_cv.shape[2] == 4:  # RGBA
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGBA2BGR)
        else:
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Dynamic resizing
        height, width = img_cv.shape[:2]
        new_width = width * scale_factor
        new_height = height * scale_factor
        
        upscaled_cv = cv2.resize(img_cv, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

        # Sharpening filter proportional to slider value
        kernel = np.array([[0, -1, 0],
                           [-1, 4 + (enhancement_power * 0.5), -1],
                           [0, -1, 0]])
        sharpened_cv = cv2.filter2D(upscaled_cv, -1, kernel)

        # Convert back to RGB for PIL
        sharpened_rgb = cv2.cvtColor(sharpened_cv, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(sharpened_rgb)

        # Apply enhancements based on slider position
        sharp_enhancer = ImageEnhance.Sharpness(result_image)
        result_image = sharp_enhancer.enhance(enhancement_power)

        contrast_enhancer = ImageEnhance.Contrast(result_image)
        result_image = contrast_enhancer.enhance(1.0 + (enhancement_power * 0.1))

    # Save to bytes for size calculation and download
    buf = BytesIO()
    result_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    res_width, res_height = result_image.size
    res_size_kb = len(byte_im) / 1024
    res_size_str = f"{res_size_kb:.2f} KB" if res_size_kb < 1024 else f"{res_size_kb/1024:.2f} MB"

    with col2:
        st.subheader("Upscaled & Enhanced")
        st.image(result_image, use_container_width=True)
        st.caption(f"📏 Resolution: {res_width} x {res_height} px\n💾 Size: {res_size_str}")

    st.success("Image updated live based on slider position!")
    
    st.download_button(
        label="Download Enhanced Image",
        data=byte_im,
        file_name="enhanced_image.png",
        mime="image/png"
    )