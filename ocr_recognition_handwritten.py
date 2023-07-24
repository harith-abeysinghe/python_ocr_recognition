import numpy as np
import cv2
from PIL import Image

# Load the image containing digits
image = cv2.imread("digits1.png")

# Convert the image to grayscale
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Divide the grayscale image into a grid of 50x100 cells (divisions)
divisions = list(np.hsplit(gray_img, 100) for i in np.vsplit(gray_img, 50))

# Convert the list of divisions into a NumPy array
NP_array = np.array(divisions)

# Prepare training and test data from the divisions
train_data = NP_array[:, :50].reshape(-1, 400).astype(np.float32)  # Using the first 50 divisions for training
test_data = NP_array[:, 50:100].reshape(-1, 400).astype(np.float32)  # Using the last 50 divisions for testing

# Open the original image again using PIL (Python Imaging Library)
image = Image.open("digits1.png")

# Create an array of labels from 0 to 9, repeated 250 times (as there are 250 samples for each digit)
k = np.arange(10)
train_labels = np.repeat(k, 250)[:, np.newaxis]
test_labels = np.repeat(k, 250)[:, np.newaxis]

# Create a k-Nearest Neighbors classifier object
knn = cv2.ml.KNearest_create()

# Train the k-NN classifier with the training data and labels
knn.train(train_data, cv2.ml.ROW_SAMPLE, train_labels)

# Find the nearest neighbors and predict the output for the test data
# 'k=3' specifies that the algorithm will consider the 3 nearest neighbors for classification
ret, output, neighbors, distance = knn.findNearest(test_data, k=3)

# Check which predictions match the true test labels
matched = output == test_labels

# Count the number of correct predictions
correct_OP = np.count_nonzero(matched)

# Calculate the accuracy of the classifier
accuracy = (correct_OP * 100) / output.size

# Print the accuracy
print(accuracy)
