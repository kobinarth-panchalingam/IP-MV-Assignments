import cv2
import os

image_path = './sample.png'
index_no = 'sample'

# Load the image from the disk
image = cv2.imread(image_path)

# Convert image to grayscale
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# create a folder named output in the current directory
if not os.path.exists('output'):
    os.makedirs('output')

# Save gray scale image
cv2.imwrite(f'./output/{index_no}_org.png', image)

# Rotate the image by 45 degrees around the center
(height, width) = image.shape[:2]
center = (width // 2, height // 2)
angle = 45
scale = 1.0
matrix = cv2.getRotationMatrix2D(center, angle, scale)
image = cv2.warpAffine(image, matrix, (width, height))

# Save rotated image
cv2.imwrite(f'./output/{index_no}_rotated.png', image)

# Create image pyramid with 5 levels
image_pyramid = [image]
for i in range(4):
    image_pyramid.append(cv2.pyrDown(image_pyramid[-1]))

# Save image pyramid
for i in range(5):
    cv2.imwrite(f'./output/{index_no}_pyramid_{i}.png', image_pyramid[i])

# Take 3rd level of image pyramid and magnify it by 4 times using bilinear interpolation
image_magnified = cv2.resize(image_pyramid[2], None, fx=4, fy=4, interpolation=cv2.INTER_LINEAR)

# Save magnified image
cv2.imwrite(f'./output/{index_no}_mag.png', image_magnified)

# Compute difference between original image and magnified image
image_diff = cv2.absdiff(image, image_magnified)

# Save difference image
cv2.imwrite(f'./output/{index_no}_diff.png', image_diff)
# The difference image is a binary image that shows the difference between the original and magnified image.
# It shows the regions where the pixel values are different between the two images.
# The difference image is black where the pixel values are the same and white where the pixel values are different.
# The difference image is useful for identifying the regions where the magnified image differs from the original image.
# The difference image can be used to identify the regions where the magnification process has introduced artifacts or errors.