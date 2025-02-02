from imutils import contours
from skimage import measure
import numpy as np
import imutils
import cv2

def preprocess_image(image_path):
    """Load the image, convert to grayscale, and apply Gaussian blur."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    return image, gray, blurred

def apply_threshold(blurred):
    """Apply binary thresholding followed by morphological operations."""
    thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)  # Reduced dilation
    return thresh

def filter_labels(thresh):
    """Label connected components and keep only significant regions."""
    labels = measure.label(thresh, connectivity=2, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    for label in np.unique(labels):
        if label == 0:
            continue

        label_mask = np.zeros(thresh.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)

        if num_pixels > 300:  # Keep only large components
            mask = cv2.add(mask, label_mask)
    
    return mask

def detect_and_draw_contours(image, mask):
    """Find contours, sort them, and draw circles with labels."""
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    for i, c in enumerate(cnts):
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
        cv2.putText(image, f"#{i + 1}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def main():
    image_path = 'bulbs.jpg'
    image, gray, blurred = preprocess_image(image_path)
    thresh = apply_threshold(blurred)
    mask = filter_labels(thresh)
    detect_and_draw_contours(image, mask)

    cv2.imshow("Detected Bulbs", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()