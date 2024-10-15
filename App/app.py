import pickle
import cv2
import numpy as np
import streamlit as st

# Define the Cartoonizer class here
class Cartoonizer:
    """Cartoonizer effect
    A class that applies a cartoon effect to an image.
    The class uses a bilateral filter and adaptive thresholding to create
    a cartoon effect.
    """
    def __init__(self, downsample_steps=2, bilateral_filters=50):
        self.downsample_steps = downsample_steps
        self.bilateral_filters = bilateral_filters

    def render(self, img_rgb):
        img_rgb = cv2.resize(img_rgb, (1366, 768))  # Resize to a fixed size
        numDownSamples = self.downsample_steps
        numBilateralFilters = self.bilateral_filters

        # -- STEP 1 -- Downsample image using Gaussian pyramid
        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        # Repeatedly apply small bilateral filter
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 9, 9, 7)

        # Upsample image to original size
        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        # -- STEPS 2 and 3 -- Convert to grayscale and apply median blur
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 3)

        # -- STEP 4 -- Detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 2)

        # -- STEP 5 -- Convert back to color for bitwise AND with color image
        (x, y, z) = img_color.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        return cv2.bitwise_and(img_color, img_edge)

    def process_image(self, img):
        cartoonized_image = self.render(img)
        return cartoonized_image

def main():
    # Page configuration
    st.set_page_config(
        page_title="Cartoonizer App",
        page_icon="🌟"
    )

    # Background styling
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://steamuserimages-a.akamaihd.net/ugc/1021696737982505623/11DB6E87522C0CEECF7DBC4AB51C4381431FF199/");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    # External CSS
    with open("Assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # App title and description
    st.title("✨ Transform Your Photos into Cartoons!")
    st.markdown("Unleash your creativity with our powerful Cartoonizer tool. Simply upload a photo and watch it magically turn into a vibrant, hand-drawn cartoon in seconds!")

    # File uploader
    uploaded_file = st.file_uploader("Upload an image to get started (JPEG or PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file to a NumPy array
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_rgb = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Create an instance of Cartoonizer
        cartoonizer = Cartoonizer()

        # Process the uploaded image
        result_image = cartoonizer.process_image(img_rgb)
        
        # Display the cartoonized image
        st.image(result_image, channels="BGR", caption="Cartoonized Image")

        # Convert the cartoonized image to PNG format for download
        _, result_image_encoded = cv2.imencode('.png', result_image)

        # Provide a download button for the cartoonized image
        st.download_button(
            label="Download Cartoonized Image",
            data=result_image_encoded.tobytes(),
            file_name="cartoonized_image.png",
            mime="image/png"
        )

if __name__ == '__main__':
    main()


