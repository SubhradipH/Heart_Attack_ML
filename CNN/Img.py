import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_ecg(image_path):
    """ Load ECG image and remove the grid. """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Use adaptive thresholding to remove the grid
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 15, 3)

    # Perform morphological operations to enhance waveforms
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned

def extract_segments(cleaned_image):
    """ Detect and extract individual ECG leads. """
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by their Y-axis (to get ECG leads in order)
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[1])

    ecg_segments = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 200 and h > 50:  # Ignore very small noise
            segment = cleaned_image[y:y+h, x:x+w]
            ecg_segments.append(segment)

    return ecg_segments

def divide_into_12_parts(image):
    """Divide the cleaned ECG image into 12 equal segments."""
    height, width = image.shape  # Get image dimensions
    segment_height = height // 12  # Calculate height of each part
    
    segments = []
    for i in range(12):
        start_y = i * segment_height
        end_y = (i + 1) * segment_height
        segment = image[start_y:end_y, :]  # Extract segment
        segments.append(segment)
    
    return segments

# Load and preprocess the ECG image
image_path = r"D:\H.A\ECG\Normal Person ECG Images (284x12=3408)-20250312T175924Z-001\Normal Person ECG Images (284x12=3408)\Normal(1).jpg"
cleaned_image = preprocess_ecg(image_path)

# Extract ECG segments
ecg_leads = extract_segments(cleaned_image)

# Divide extracted ECG into 12 parts
ecg_leads_12 = divide_into_12_parts(cleaned_image)

# Display the 12 ECG lead images
plt.figure(figsize=(10, 8))
for i, segment in enumerate(ecg_leads_12):
    plt.subplot(4, 3, i + 1)  # Arrange in a 4x3 grid
    plt.imshow(segment, cmap="gray")
    plt.axis("off")
    plt.title(f"Lead {i+1}")
plt.suptitle("12-Lead ECG Segments")
plt.show()
