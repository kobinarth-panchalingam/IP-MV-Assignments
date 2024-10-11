import cv2
import numpy as np
import os

# create a folder named output in the current directory
if not os.path.exists('output'):
    os.makedirs('output')

def read_image(file_path):
    # Read the image using OpenCV
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    return image

def save_image(file_path, image):
    # Save the image using OpenCV
    cv2.imwrite(file_path, image)

def mean_filter(image, kernel_size):
    # Create a mean filter kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)
    result = np.zeros_like(image, dtype=np.float32)
    pad = kernel_size // 2

    # Apply mean filter to the image
    for i in range(pad, image.shape[0] - pad):
        for j in range(pad, image.shape[1] - pad):
            result[i, j] = np.sum(image[i-pad:i+pad+1, j-pad:j+pad+1] * kernel)

    return result.astype(np.uint8)

def median_filter(image, kernel_size):
    result = np.zeros_like(image, dtype=np.uint8)
    pad = kernel_size // 2

    # Apply median filter to the image
    for i in range(pad, image.shape[0] - pad):
        for j in range(pad, image.shape[1] - pad):
            result[i, j] = np.median(image[i-pad:i+pad+1, j-pad:j+pad+1])

    return result

def k_closest_averaging(image, kernel_size, k):
    result = np.zeros_like(image, dtype=np.uint8)
    pad = kernel_size // 2

    # Apply k-closest averaging filter to the image
    for i in range(pad, image.shape[0] - pad):
        for j in range(pad, image.shape[1] - pad):
            values = image[i-pad:i+pad+1, j-pad:j+pad+1].ravel()
            values.sort()
            result[i, j] = np.mean(values[:k])

    return result

def threshold_averaging(image, kernel_size, threshold):
    result = np.zeros_like(image, dtype=np.uint8)
    pad = kernel_size // 2

    # Apply threshold averaging filter to the image
    for i in range(pad, image.shape[0] - pad):
        for j in range(pad, image.shape[1] - pad):
            values = image[i-pad:i+pad+1, j-pad:j+pad+1].ravel()
            mean_value = np.mean(values)
            result[i, j] = np.mean([v for v in values if abs(v - mean_value) < threshold])

    return result


# Example usage:
input_image = read_image('./sample_image.png')
input_image_2 = read_image('./sample_image2.png')

# Mean filter with kernel size 3, 5 and 7
mean_filtered_image_3 = mean_filter(input_image_2, kernel_size=3)
mean_filtered_image_5= mean_filter(input_image_2, kernel_size=5)
mean_filtered_image_7 = mean_filter(input_image_2, kernel_size=7)

# Median filter with kernel size 3, 5 and 7
median_filtered_image_3 = median_filter(input_image, kernel_size=3)
median_filtered_image_5 = median_filter(input_image, kernel_size=5)
median_filtered_image_7 = median_filter(input_image, kernel_size=7)

# k-closest averaging with kernel size 5 and k=10,20
k_closest_filtered_image_5_10 = k_closest_averaging(input_image_2, kernel_size=5, k=10)
k_closest_filtered_image_5_20 = k_closest_averaging(input_image_2, kernel_size=5, k=20)

# Threshold averaging with kernel size 3 and threshold value 20 and 100
threshold_filtered_image_3_20 = threshold_averaging(input_image_2, kernel_size=3, threshold=20)
threshold_filtered_image_3_100 = threshold_averaging(input_image_2, kernel_size=3, threshold=100)

# Save filtered images
save_image('./output/mean_filtered_image_3.jpg', mean_filtered_image_3)
save_image('./output/mean_filtered_image_5.jpg', mean_filtered_image_5)
save_image('./output/mean_filtered_image_7.jpg', mean_filtered_image_7)

save_image('./output/median_filtered_image_3.jpg', median_filtered_image_3)
save_image('./output/median_filtered_image_5.jpg', median_filtered_image_5)
save_image('./output/median_filtered_image_7.jpg', median_filtered_image_7)

save_image('./output/k_closest_filtered_image_5_10.jpg', k_closest_filtered_image_5_10)
save_image('./output/k_closest_filtered_image_5_20.jpg', k_closest_filtered_image_5_20)

save_image('./output/threshold_filtered_image_3_20.jpg', threshold_filtered_image_3_20)
save_image('./output/threshold_filtered_image_3_100.jpg', threshold_filtered_image_3_100)