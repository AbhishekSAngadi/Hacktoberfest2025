import cv2
import numpy as np

def enhance_image_with_opencv(input_path, output_path):
    """
    Enhances an image using unsharp masking with OpenCV.

    Args:
        input_path (str): The path to the input image file.
        output_path (str): The path to save the enhanced image.
    """
    try:
        # Read the image
        img = cv2.imread(input_path)
        if img is None:
            raise FileNotFoundError(f"Image not found at {input_path}")

        # Convert to grayscale for easier processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply a Gaussian blur to create a soft version of the image
        blurred = cv2.GaussianBlur(gray, (0, 0), 3)

        # Apply unsharp masking by subtracting the blurred image from the original
        # The scale factors control the intensity of the sharpening
        sharpened = cv2.addWeighted(gray, 1.5, blurred, -0.5, 0)

        # Save the result
        cv2.imwrite(output_path, sharpened)
        print(f"Image saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Example usage
    enhance_image_with_opencv('input_image.jpg', 'sharpened_image.jpg')
