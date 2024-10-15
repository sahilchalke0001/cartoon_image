import cv2
import pickle

class Cartoonizer:
    """Cartoonizer effect
    A class that applies a cartoon effect to an image.
    The class uses a bilateral filter and adaptive thresholding to create
    a cartoon effect.
    """
    def __init__(self, downsample_steps=2, bilateral_filters=50):
        self.downsample_steps = downsample_steps
        self.bilateral_filters = bilateral_filters

    def render(self, img_path):
        # Load the image
        img_rgb = cv2.imread(img_path)
        img_rgb = cv2.resize(img_rgb, (1366, 768))  # Resize to a fixed size
        numDownSamples = self.downsample_steps  # Number of downscaling steps
        numBilateralFilters = self.bilateral_filters  # Number of bilateral filtering steps

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
        img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)  # Use COLOR_BGR2GRAY
        img_blur = cv2.medianBlur(img_gray, 3)

        # -- STEP 4 -- Detect and enhance edges
        img_edge = cv2.adaptiveThreshold(img_blur, 255,
                                          cv2.ADAPTIVE_THRESH_MEAN_C,
                                          cv2.THRESH_BINARY, 9, 2)

        # -- STEP 5 -- Convert back to color for bitwise AND with color image
        (x, y, z) = img_color.shape
        img_edge = cv2.resize(img_edge, (y, x))
        img_edge = cv2.cvtColor(img_edge, cv2.COLOR_GRAY2RGB)

        # Save the edge image (optional)
        cv2.imwrite("edge.png", img_edge)

        # Combine color and edge images
        return cv2.bitwise_and(img_color, img_edge)

    def process_image(self, img_path, output_path):
        # Process the image and save the cartoonized version
        cartoonized_image = self.render(img_path)
        cv2.imwrite(output_path, cartoonized_image)
        return cartoonized_image

    def show_image(self, img):
        # Display the image
        cv2.imshow("Cartoon version", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Save the Cartoonizer instance as a pickle file
if __name__ == "__main__":
    cartoonizer = Cartoonizer()
    with open('cartoonizer_model.pkl', 'wb') as f:
        pickle.dump(cartoonizer, f)



 
