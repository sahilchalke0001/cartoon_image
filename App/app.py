import pickle
import time
import math
import cv2
import numpy as np
import streamlit as st

class Cartoonizer:
    """Cartoonizer effect
    A class that applies a cartoon-like black and white effect to an image.
    The class uses edge detection and adaptive thresholding to create
    a cartoon-styled image.
    """
    def __init__(self, downsample_steps=1, bilateral_filters=1):
        self.downsample_steps = downsample_steps
        self.bilateral_filters = bilateral_filters

    def render(self, img_rgb):
        numDownSamples = self.downsample_steps
        numBilateralFilters = self.bilateral_filters

        img_color = img_rgb
        for _ in range(numDownSamples):
            img_color = cv2.pyrDown(img_color)

        # Use a smaller diameter in the bilateral filter to retain more detail
        for _ in range(numBilateralFilters):
            img_color = cv2.bilateralFilter(img_color, 7, 50, 50)

        for _ in range(numDownSamples):
            img_color = cv2.pyrUp(img_color)

        img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        # Reduce blur to keep finer details
        img_blur = cv2.medianBlur(img_gray, 1)

        # Use a smaller blockSize and adjust the C value for thinner outlines
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 3, 1)

        return img_edge

    def process_image(self, img):
        cartoon_image = self.render(img)
        return cartoon_image

def main():
    st.set_page_config(
        page_title="Cartoonizer App",
        page_icon="🎨"
    )

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

    st.title("✨ Turn Your Photos into Cartoon Art!")
    st.markdown("Transform your photos with our Cartoonizer tool. Upload a photo to generate a cartoon-style, black-and-white image with striking outlines and minimal noise.")

    uploaded_file = st.file_uploader("Upload an image to get started (JPEG or PNG)...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            img_rgb = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            cartoonizer = Cartoonizer()
            result_image = cartoonizer.process_image(img_rgb)

            st.image(result_image, channels="GRAY", caption="Cartoon-styled Image")

            _, result_image_encoded = cv2.imencode('.png', result_image)
            timestamp = math.floor(time.time() * 1000000)
            st.download_button(
                label="Download Cartoon-styled Image",
                data=result_image_encoded.tobytes(),
                file_name=f'{timestamp}.png',
                mime="image/png"
            )
        except Exception as e:
            st.error("There was an error processing the image. Please try a different image.")
            st.error(f"Error details: {e}")

if __name__ == '__main__':
    main()


