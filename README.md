# SEC-DIP--Coin-Detection-using-OpenCV-in-Python
## Aim
In this project, you will work with an image and apply morphological operations and thresholding techniques to detect and count the total number of coins present in the image.

## Program
```
import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Step 1: Read Image
# -----------------------------
# Read image
# Store it in the variable image
###
### YOUR CODE HERE
###
image = cv2.imread('coins.jpg')  # <- change filename if needed
assert image is not None, "Image not found. Place 'coins.jpg' in the working directory or change the filename."

# Dont Change the Code 
imageCopy = image.copy()
plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off'); plt.show()
# Expected output: Original image in color


# -----------------------------
# Step 2: Convert Image to Grayscale
# -----------------------------
# Convert image to grayscale
# Store it in the variable imageGray
###
### YOUR CODE HERE
###
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.figure(figsize=(12,12))
plt.subplot(121); plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off')
plt.subplot(122); plt.imshow(imageGray, cmap='gray'); plt.title("Grayscale Image"); plt.axis('off'); plt.show()
# Notes: Grayscale simplifies intensity-based ops; coins typically appear bright vs darker background.


# -----------------------------
# Step 3: Split Image into R,G,B Channels
# -----------------------------
# Split image into channels
# Store them in variables imageB, imageG, imageR
###
### YOUR CODE HERE
###
imageB, imageG, imageR = cv2.split(image)

plt.figure(figsize=(20,12))
plt.subplot(141); plt.imshow(image[:,:,::-1]); plt.title("Original Image"); plt.axis('off')
plt.subplot(142); plt.imshow(imageB, cmap='gray'); plt.title("Blue Channel"); plt.axis('off')
plt.subplot(143); plt.imshow(imageG, cmap='gray'); plt.title("Green Channel"); plt.axis('off')
plt.subplot(144); plt.imshow(imageR, cmap='gray'); plt.title("Red Channel"); plt.axis('off')
plt.show()
# Notes: Depending on lighting, one channel may give better contrast between coins and background.


# -----------------------------
# Step 4: Perform Thresholding
# -----------------------------
# Try multiple thresholds; keep all intermediate images and note observations.
###
### YOUR CODE HERE
###
# 1) Otsu's threshold on grayscale (robust automatic)
_, otsuBin = cv2.threshold(imageGray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# 2) Fixed thresholds to compare
fixed_thresholds = [80, 100, 120, 150]
fixed_bins = []
for t in fixed_thresholds:
    _, bin_img = cv2.threshold(imageGray, t, 255, cv2.THRESH_BINARY)
    fixed_bins.append((t, bin_img))

# Display the thresholded images
###
### YOUR CODE HERE
###
plt.figure(figsize=(16,10))
plt.subplot(231); plt.imshow(imageGray, cmap='gray'); plt.title("Grayscale"); plt.axis('off')
plt.subplot(232); plt.imshow(otsuBin, cmap='gray'); plt.title("Otsu Binary"); plt.axis('off')
for i, (t, bin_img) in enumerate(fixed_bins, start=3):
    plt.subplot(2,3,i); plt.imshow(bin_img, cmap='gray'); plt.title(f"Fixed Thresh = {t}"); plt.axis('off')
plt.tight_layout(); plt.show()

# Choose a working binary for morphology. Often Otsu works well; if background is white and coins dark, invert.
# Check foreground polarity: If coins are bright on dark, keep; if coins are dark on bright, invert.
# Heuristic: count mean intensity of coins region is unknown; we assume coins are brighter -> keep as is.
binary = otsuBin.copy()

# If your coins appear black and background white, invert:
# binary = cv2.bitwise_not(otsuBin)


# -----------------------------
# Step 5: Perform morphological operations
# -----------------------------
# Try different kernel sizes/shapes and operations. Keep intermediates and findings.
###
### YOUR CODE HERE
###
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

# Opening to remove small noise
imageOpened_3 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1, iterations=1)
imageOpened_5 = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel2, iterations=1)

# Closing to fill small holes within coins
imageClosed_3 = cv2.morphologyEx(imageOpened_5, cv2.MORPH_CLOSE, kernel1, iterations=1)
imageClosed_5 = cv2.morphologyEx(imageOpened_5, cv2.MORPH_CLOSE, kernel2, iterations=1)

###
### YOUR CODE HERE
###
# Dilation (helps connect broken coin edges)
imageDilated1 = cv2.dilate(imageClosed_5, kernel2, iterations=1)
imageDilated2 = cv2.dilate(imageClosed_5, kernel2, iterations=2)

plt.imshow(imageDilated2, cmap='gray'); plt.title('Dilated Image Iteration 2'); plt.axis('off'); plt.show()
# Observation: 2 iterations often ensure coin blobs are solid without fusing neighboring coins (tune as needed).


# Display all the intermediate step images for documentation
###
### YOUR CODE HERE
###
plt.figure(figsize=(16,14))
plt.subplot(241); plt.imshow(binary, cmap='gray'); plt.title("Chosen Binary"); plt.axis('off')
plt.subplot(242); plt.imshow(imageOpened_3, cmap='gray'); plt.title("Opening (3x3)"); plt.axis('off')
plt.subplot(243); plt.imshow(imageOpened_5, cmap='gray'); plt.title("Opening (5x5)"); plt.axis('off')
plt.subplot(244); plt.imshow(imageClosed_3, cmap='gray'); plt.title("Closing (3x3)"); plt.axis('off')
plt.subplot(245); plt.imshow(imageClosed_5, cmap='gray'); plt.title("Closing (5x5)"); plt.axis('off')
plt.subplot(246); plt.imshow(imageDilated1, cmap='gray'); plt.title("Dilated x1 (5x5)"); plt.axis('off')
plt.subplot(247); plt.imshow(imageDilated2, cmap='gray'); plt.title("Dilated x2 (5x5)"); plt.axis('off')
plt.tight_layout(); plt.show()

# Get structuring element/kernel which will be used for dilation (explicit block, as asked)
###
### YOUR CODE HERE
###
kernel_for_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))

# Optional erosion to separate tiny fused blobs after dilation
###
### YOUR CODE HERE
###
imageEroded = cv2.erode(imageDilated2, kernel_for_erosion, iterations=1)

plt.imshow(imageEroded, cmap='gray'); plt.title("Eroded Image"); plt.axis('off'); plt.show()
# Note: If coins start separating incorrectly, reduce erosion iterations or skip it.


# -----------------------------
# Step 5 (given): Create SimpleBlobDetector
# -----------------------------
# Set up the SimpleBlobdetector with default parameters.
params = cv2.SimpleBlobDetector_Params()

params.blobColor = 0
params.minDistBetweenBlobs = 2

# Filter by Area.
params.filterByArea = False  # (let morphology shape the blobs first)

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.8

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.8

# Create SimpleBlobDetector
detector = cv2.SimpleBlobDetector_create(params)


# -----------------------------
# Step 6: Detect Coins (Blob detection)
# -----------------------------
# Detect blobs
###
### YOUR CODE HERE
###
# Ensure correct polarity for blobColor:
# blobColor=0 -> detector looks for dark blobs. If our coins are white blobs on black background, invert:
pre_blob = imageEroded.copy()
if params.blobColor == 0:
    # If foreground is white, invert so blobs become dark
    white_ratio = pre_blob.mean() / 255.0
    if white_ratio > 0.5:
        pre_blob = cv2.bitwise_not(pre_blob)

keypoints = detector.detect(pre_blob)

# Print number of coins detected
###
### YOUR CODE HERE
###
print(f"Number of coins detected (Blob): {len(keypoints)}")

# Visualize keypoints
im_with_keypoints = cv2.drawKeypoints(imageCopy, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(im_with_keypoints[:,:,::-1]); plt.title(f"Blob Detections: {len(keypoints)}"); plt.axis('off'); plt.show()

# Expected output
# Number of coins detected: 9 (this depends on your image)


# -----------------------------
# BONUS: Detect Coins using Contours (as required)
# -----------------------------
# Find contours on a clean binary (use imageEroded). Filter by area and circularity.
cnts, _ = cv2.findContours(imageEroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def contour_circularity(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    if peri == 0:
        return 0.0
    # Circularity = 4πA / P^2 ; close to 1 for circles
    return (4.0 * np.pi * area) / (peri * peri)

contours_kept = []
areas = []
for c in cnts:
    area = cv2.contourArea(c)
    circ = contour_circularity(c)
    # Heuristic filters: tweak as needed depending on image scale
    if area > 100 and circ > 0.65:
        contours_kept.append(c)
        areas.append(area)

contour_vis = imageCopy.copy()
cv2.drawContours(contour_vis, contours_kept, -1, (0,255,0), 2)
print(f"Number of coins detected (Contours): {len(contours_kept)}")

plt.imshow(contour_vis[:,:,::-1]); plt.title(f"Contour Detections: {len(contours_kept)}"); plt.axis('off'); plt.show()

# -----------------------------
# Final Notes for Report (copy into your submission):
# - Grayscale simplifies intensity analysis; Blue/Green/Red channel inspection showed which channel had best contrast.
# - Otsu's threshold gave a stable binary; fixed thresholds demonstrate sensitivity to lighting (too low -> merges background; too high -> breaks coins).
# - Morphology:
#     * Opening (3x3, 5x5) removed salt noise.
#     * Closing (3x3, 5x5) filled small holes within coin regions.
#     * Dilation (x1, x2) strengthened coin blobs; over-dilation may fuse adjacent coins.
#     * Light erosion after dilation helped separate minor fusions; excessive erosion can break coins.
# - Blob vs Contour:
#     * Blob detection is robust if polarity matches blobColor; simple and tunable via shape filters (circularity, inertia).
#     * Contour detection with circularity and area thresholds is explainable and often precise on clean binaries.
# - Always keep intermediate images to justify chosen parameters for your specific image.
```
## Output
<img width="448" height="510" alt="image" src="https://github.com/user-attachments/assets/f88b834c-b786-49fb-b9aa-ccfd31f5922b" />
<img width="448" height="510" alt="image" src="https://github.com/user-attachments/assets/c591666e-4df3-4c05-9105-0bd73701054d" />

## Result
Thus the program to detect the edges was executed successfully.
