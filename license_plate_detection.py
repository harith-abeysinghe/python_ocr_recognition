import imutils
import pytesseract
from PIL import Image
import cv2

# Load the input image
image = cv2.imread("test3.png")

# Resize the image to a width of 300 pixels (keeping aspect ratio)
image = imutils.resize(image, width=300)

# Display the original image
cv2.imshow("Original Image", image)
cv2.waitKey(0)

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display the grayscale image
cv2.imshow("Gray Image", gray_image)
cv2.waitKey(0)

# Apply bilateral filtering for noise reduction while preserving edges
gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17, 17)

# Display the smoothed image
cv2.imshow("Smoothed Image", gray_image)
cv2.waitKey(0)

# Apply Canny edge detection to find edges in the image
edged = cv2.Canny(gray_image, 30, 200)

# Display the edged image
cv2.imshow("Edged", edged)
cv2.waitKey(0)

# Find contours in the edged image
cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Create a copy of the original image and draw all contours found
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)

# Display the image with all contours
cv2.imshow("Contours", image1)
cv2.waitKey(0)

# Sort the contours based on their areas in descending order and keep the top 30
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]

# Initialize variable to store the detected license plate contour
screenCnt = None

# Create a copy of the original image and draw the top 30 contours
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)

# Display the image with the top 30 contours
cv2.imshow("Top 30 Contours", image2)
cv2.waitKey(0)

# Loop through the top 30 contours to find the license plate contour (rectangle with 4 corners)
i = 7
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c)
        # Crop the license plate region from the original image
        new_img = image[y:y + h, x:x + w]
        # Save the cropped license plate as a new image
        cv2.imwrite('./' + str(i) + '.png', new_img)
        i += 1
        break

# Draw the detected license plate contour on the original image
cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)

# Display the original image with the detected license plate contour
cv2.imshow("Detected License Plate", image)
cv2.waitKey(0)

# Path to the cropped license plate image
Cropped_loc = './7.png'

# Read the cropped license plate image using pytesseract to extract text from it
plate = pytesseract.image_to_string(Cropped_loc, lang='eng')

# Print the extracted text (number plate)
print("Number plate is:", plate)

cv2.waitKey(0)
cv2.destroyAllWindows()
