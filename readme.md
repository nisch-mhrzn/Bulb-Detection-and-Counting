# Bulb Detection and Counting

## Project Overview
This project detects and counts illuminated bulbs in an image using OpenCV and the scikit-image library. It applies image processing techniques such as grayscale conversion, Gaussian blurring, thresholding, morphological operations, and contour detection to identify individual bulbs.

## Features
- Converts the image to grayscale for better processing
- Applies Gaussian blur to reduce noise
- Uses thresholding to segment the illuminated bulbs
- Performs morphological operations (erosion and dilation) for better segmentation
- Labels and filters connected components to isolate bulbs
- Finds and sorts contours of detected bulbs
- Draws enclosing circles around detected bulbs and labels them with index numbers

## Prerequisites
Ensure you have the following libraries installed:

```bash
pip install opencv-python numpy imutils scikit-image
```

## Usage
1. Place the `bulbs.jpg` image in the project directory.
2. Run the script using:

```bash
python main.py
```

3. The processed image will be displayed with detected bulbs marked and numbered.

## Code Explanation
1. **Load the Image**: Reads the image and converts it to grayscale.
2. **Preprocessing**: Applies Gaussian blur and thresholding to segment the bulbs.
3. **Morphological Operations**: Uses erosion and dilation to refine the segmentation.
4. **Labeling & Masking**: Identifies connected components and filters out small regions.
5. **Contour Detection**: Extracts contours of detected bulbs.
6. **Drawing & Labeling**: Draws enclosing circles and labels each detected bulb.

## Output Example
The output image will display detected bulbs with red circles and numbered labels.


