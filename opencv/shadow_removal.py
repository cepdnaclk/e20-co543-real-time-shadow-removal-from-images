import cv2
import numpy as np

def remove_shadows(input_path, output_path):
    """
    Remove shadows from an image using HSV and YCbCr color space corrections.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image with shadows removed.

    Raises:
        ValueError: If the image is not found or invalid format.
    """
    # Load the image
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]

    # Threshold to create a shadow mask
    _, shadow_mask = cv2.threshold(v_channel, 80, 255, cv2.THRESH_BINARY_INV)

    # Remove noise using morphological operations
    kernel = np.ones((5, 5), np.uint8)
    shadow_mask = cv2.morphologyEx(shadow_mask, cv2.MORPH_CLOSE, kernel)

    # Identify shadow and non-shadow regions
    shadow_pixels = shadow_mask == 255
    avg_non_shadow = np.mean(v_channel[~shadow_pixels])

    # Adjust shadow pixels' brightness
    v_channel[shadow_pixels] = np.clip(v_channel[shadow_pixels] * (255 / avg_non_shadow), 0, 255)

    # Update the HSV image and convert back to BGR
    hsv[:, :, 2] = v_channel
    corrected_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Convert to YCbCr for further intensity correction
    ycbcr = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2YCrCb)
    y_channel = ycbcr[:, :, 0]

    # Calculate mean and standard deviation for intensity correction
    y_mean = np.mean(y_channel)
    y_std = np.std(y_channel)

    # Refine shadow mask
    refined_shadow_mask = np.where(y_channel < y_mean - (y_std / 3), 255, 0).astype(np.uint8)

    # Further enhance brightness in shadow regions
    avg_intensity_lit = np.mean(y_channel[refined_shadow_mask == 0])
    avg_intensity_shadow = np.mean(y_channel[refined_shadow_mask == 255])
    intensity_difference = avg_intensity_lit - avg_intensity_shadow

    y_channel[refined_shadow_mask == 255] = np.clip(y_channel[refined_shadow_mask == 255] + intensity_difference, 0, 255)
    ycbcr[:, :, 0] = y_channel

    # Convert back to BGR and save the output
    final_result = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(output_path, final_result)
