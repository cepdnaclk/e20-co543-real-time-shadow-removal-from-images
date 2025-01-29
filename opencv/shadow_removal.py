import cv2
import numpy as np


def remove_shadows(input_path, output_path):
    """
    Remove shadows from an image.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed image with shadows removed.

    Raises:
        ValueError: If the image is not found or invalid format.
    """
    # Reload the original image to ensure no prior modifications affect it
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")

    # Convert the image to LAB color space to better isolate shadows
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance the L channel
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)

    # Merge back the LAB channels
    enhanced_lab = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # Mask shadow regions by detecting darker areas compared to their surroundings
    shadow_mask_refined = cv2.inRange(enhanced_l_channel, 0, 120)

    # Inpainting to remove shadows only
    final_inpainted = cv2.inpaint(image, shadow_mask_refined, inpaintRadius=7, flags=cv2.INPAINT_TELEA)

    # Save the refined result
    cv2.imwrite(output_path, final_inpainted)

