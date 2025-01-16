import cv2

def convert_to_grayscale(input_path, output_path):
    """
    Convert an image to grayscale.

    Args:
        input_path (str): Path to the input image.
        output_path (str): Path to save the processed grayscale image.

    Raises:
        ValueError: If the image is not found or invalid format.
    """
    image = cv2.imread(input_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(output_path, gray_image)
