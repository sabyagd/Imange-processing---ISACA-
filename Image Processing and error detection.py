#!/usr/bin/env python
# coding: utf-8

# In[89]:


import cv2
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('silicon-wafer-orig.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(histogram, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram Analysis')
plt.show()


# In[90]:


# Load the image
image = cv2.imread('silicon-wafer-bright.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(histogram, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram Analysis')
plt.show()


# In[91]:


# Load the image
image = cv2.imread('silicon-wafer-contrast.jpg', cv2.IMREAD_GRAYSCALE)

# Compute the histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

# Plot the histogram
plt.plot(histogram, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram Analysis')
plt.show()


# In[5]:


# Load the image
image = cv2.imread('Flower.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('Flower_1.jpg', cv2.IMREAD_GRAYSCALE)
# Compute the histogram
histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
histogram2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
# Plot the histogram
plt.plot(histogram, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram Analysis')
plt.show()
plt.plot(histogram2, color='gray')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram Analysis')
plt.show()


# In[88]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
image1 = cv2.imread('w_orig.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('w_d1.jpg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('w_d2.jpg', cv2.IMREAD_GRAYSCALE)

# Calculate the histograms of both images
hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
hist3 = cv2.calcHist([image3], [0], None, [256], [0, 256])
#hist4 = cv2.calcHist([image4], [0], None, [256], [0, 256])

# Normalize the histograms
hist1 /= hist1.sum()
hist2 /= hist2.sum()
hist3 /= hist3.sum()
#hist4 /= hist4.sum()

# Calculate the Chi-Squared distance between the histograms
#chi_squared_distance = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

plt.figure(figsize=(15, 6))

plt.subplot(1, 3, 1)
plt.imshow(image1)
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(image2)
plt.title('Visible Faulty Image')

plt.subplot(1, 3, 3)
plt.imshow(image3)
plt.title('Major Fault Image')

plt.show()

# Print the result
print("Deffect identified Percentage for Visible Faulty Image:", "{:.4f}%".format(cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)*100))
print("Deffect identified Percentage for Major Fault Image:", "{:.4f}%".format(cv2.compareHist(hist1, hist3, cv2.HISTCMP_CHISQR)*100))


# In[87]:


import cv2
import numpy as np
from skimage.metrics import mean_squared_error

# Load the images
image1 = cv2.imread('silicon-wafer-orig.jpg', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('silicon-wafer-bright.jpg', cv2.IMREAD_GRAYSCALE)
image3 = cv2.imread('silicon-wafer-contrast.jpg', cv2.IMREAD_GRAYSCALE)

# Display original and bright images
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(image1, cmap='gray')
plt.title('Original Image')

plt.subplot(1, 3, 2)
plt.imshow(image2, cmap='gray')
plt.title('Bright Image')

plt.subplot(1, 3, 3)
plt.imshow(image3, cmap='gray')
plt.title('HIgh Contrast Image')

plt.show()

equalized_image1 = cv2.equalizeHist(image1)
equalized_image2 = cv2.equalizeHist(image2)
equalized_image3 = cv2.equalizeHist(image3)

# Get the dimensions of the images
height1, width1 = image1.shape
height2, width2 = image2.shape
height3, width3 = image3.shape

print("Dimensions of image 1: Height =", height1, "Width =", width1)
print("Dimensions of image 2: Height =", height2, "Width =", width2)
print("Dimensions of image 3: Height =", height3, "Width =", width3)

# Normalize pixel values
normalized_image1 = image1 / 255.0
normalized_image2 = image2 / 255.0
normalized_image3 = image3 / 255.0

# Compare images using Mean Squared Error (MSE)
mse_norm1 = mean_squared_error(normalized_image1, normalized_image2)

print("Normalized Mean Squared Error (MSE):", "{:.5f}".format(mse_norm1))

mse_norm2 = mean_squared_error(normalized_image1, normalized_image3)

print("Normalized Mean Squared Error (MSE):", "{:.5f}".format(mse_norm2))

# Equalized pixel values
equalized_image1 = equalized_image1 / 255.0
equalized_image2 = equalized_image2 / 255.0
equalized_image3 = equalized_image3 / 255.0


# Compare images using Mean Squared Error (MSE)
mse1 = mean_squared_error(equalized_image1, equalized_image2)

print("Equalized Mean Squared Error (MSE):", "{:.4f}".format(mse1))

mse2 = mean_squared_error(equalized_image1, equalized_image3)

print("Equalized Mean Squared Error (MSE):", "{:.4f}".format(mse2))


# In[49]:


# Load the images
image4 = cv2.imread('Flower.jpg', cv2.IMREAD_GRAYSCALE)
image5 = cv2.imread('Flower_3.jpg', cv2.IMREAD_GRAYSCALE)

# Display original and bright images
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.imshow(image4)
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(image5, cmap='gray')
plt.title('Faulty Image')

plt.show()

equalized_image4 = cv2.equalizeHist(image4)
equalized_image5 = cv2.equalizeHist(image5)

# Get the dimensions of the images
height4, width4 = image4.shape
height5, width5 = image5.shape

print("Dimensions of image 4: Height =", height4, "Width =", width4)
print("Dimensions of image 5: Height =", height5, "Width =", width5)

# Normalize pixel values
normalized_image4 = image4 / 255.0
normalized_image5 = image5 / 255.0

# Compare images using Mean Squared Error (MSE)
mse_norm = mean_squared_error(normalized_image4, normalized_image5)

print("Normalized Mean Squared Error (MSE):", "{:.4f}".format(mse_norm))

# Equalized pixel values
equalized_image4 = equalized_image4 / 255.0
equalized_image5 = equalized_image5 / 255.0

# Compare images using Mean Squared Error (MSE)
mse1 = mean_squared_error(equalized_image4, equalized_image5)

print("Equalized Mean Squared Error (MSE):", "{:.4f}".format(mse1))


# In[ ]:




