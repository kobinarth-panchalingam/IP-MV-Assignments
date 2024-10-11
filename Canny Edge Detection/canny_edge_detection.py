import numpy as np
import cv2
import sys
import os

def load_image(file_path):
    return cv2.imread(file_path)

def save_image(file_path, image):
    cv2.imwrite(file_path, image)

def convert_to_grayscale(image):
    # Calculate the grayscale value of each pixel using average of RGB values
    gray_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gray_image[i, j] = np.average(image[i, j])
    return gray_image

def apply_gaussian_filter(image, kernel_size=5, sigma=1):
    # Crate a 1-D Gaussian filter
    ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))

    # Take the outer product of the Gaussian filter with itself to get a 2D filter
    kernel = np.outer(gauss, gauss)

    # Normalize the kernel
    kernel = kernel / np.sum(kernel)

    # Pad the image to apply the convolution operation
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='constant')
    filtered_image = np.zeros_like(image)

    # Apply the convolution operation
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            filtered_image[i, j] = np.sum(kernel * padded_image[i:i + kernel_size, j:j + kernel_size])

    return filtered_image

def compute_gradients(image, sobel_kernel_size):
    if sobel_kernel_size == 3:
        # Define Sobel kernels (3x3)
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    elif sobel_kernel_size == 5:
        sobel_x = np.array([[-1, -2, 0, 2, 1], [-2, -3, 0, 3, 2], [-3, -5, 0, 5, 3], [-2, -3, 0, 3, 2], [-1, -2, 0, 2, 1]])
        sobel_y = np.array([[-1, -2, -3, -2, -1], [-2, -3, -5, -3, -2], [0, 0, 0, 0, 0], [2, 3, 5, 3, 2], [1, 2, 3, 2, 1]])
    else:
        sys.exit("Sobel kernel size should be 3 or 5!")

    # Get the dimensions of the image
    rows, cols = image.shape

    # Initialize arrays for gradient magnitude and orientation
    gradient_x = np.zeros_like(image, dtype=np.float64)
    gradient_y = np.zeros_like(image, dtype=np.float64)

    # Compute gradient using Sobel operators
    half_size = sobel_kernel_size // 2
    for i in range(half_size, rows - half_size):
        for j in range(half_size, cols - half_size):
            window = image[i - half_size:i + half_size + 1, j - half_size:j + half_size + 1]
            gradient_x[i, j] = np.sum(window * sobel_x)
            gradient_y[i, j] = np.sum(window * sobel_y)

    # Compute gradient magnitude and orientation
    magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
    orientation = np.arctan2(gradient_y, gradient_x)

    return magnitude, orientation

def non_maximum_suppression(image, theta):
    M, N = image.shape
    suppressed_image = np.zeros((M, N), dtype=np.int32)
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M-1):
        for j in range(1, N-1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = image[i, j+1]
                    r = image[i, j-1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = image[i+1, j-1]
                    r = image[i-1, j+1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = image[i+1, j]
                    r = image[i-1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = image[i-1, j-1]
                    r = image[i+1, j+1]

                if (image[i, j] >= q) and (image[i, j] >= r):
                    suppressed_image[i, j] = image[i, j]
                else:
                    suppressed_image[i, j] = 0

            except IndexError as e:
                pass

    return suppressed_image

def threshold(image, lowThreshold, highThreshold):
    M, N = image.shape
    thresholded_image = np.zeros((M, N), dtype=np.int32)

    # Define weak and strong pixels
    weak = np.int32(25)
    strong = np.int32(255)

    # Create arrays to hold weak and strong pixel locations
    strong_i, strong_j = np.where(image >= highThreshold)
    zeros_i, zeros_j = np.where(image < lowThreshold)
    weak_i, weak_j = np.where((image <= highThreshold) & (image >= lowThreshold))

    # Set the values of weak and strong pixels
    thresholded_image[strong_i, strong_j] = strong
    thresholded_image[weak_i, weak_j] = weak

    return thresholded_image, weak, strong

def hysteresis(image, weak, strong=255):
    M, N = image.shape
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (image[i, j] == weak):
                try:
                    # Check 8-connected neighbors
                    if ((image[i+1, j-1] == strong) or (image[i+1, j] == strong) or (image[i+1, j+1] == strong)
                        or (image[i, j-1] == strong) or (image[i, j+1] == strong)
                        or (image[i-1, j-1] == strong) or (image[i-1, j] == strong) or (image[i-1, j+1] == strong)):
                        image[i, j] = strong
                    else:
                        image[i, j] = 0
                except IndexError as e:
                    pass
    return image

def canny_edge_detection(image_path):
    image = load_image(image_path)
    save_image("1_original.png", image)
    
    gray_image = convert_to_grayscale(image)
    save_image("2_grayscale.png", gray_image)

    kernel_size = 5
    sigma = 1
    blurred_image = apply_gaussian_filter(gray_image, kernel_size, sigma)
    save_image("3_blurred.png", blurred_image)
    
    sobel_kernel_size = 3
    gradient_magnitude, gradient_direction = compute_gradients(blurred_image, sobel_kernel_size)
    save_image("4_gradient_magnitude.png", gradient_magnitude)
    
    non_max_suppressed_image = non_maximum_suppression(gradient_magnitude, gradient_direction)
    save_image("5_non_max_suppressed.png", non_max_suppressed_image)
    
    low_threshold = 30
    high_threshold = 120
    thresholded_image, weak, strong = threshold(non_max_suppressed_image,low_threshold, high_threshold)
    save_image("6_thresholded.png", thresholded_image)
    
    final_image = hysteresis(thresholded_image, weak, strong)
    
    base_name = os.path.splitext(image_path)[0]
    output_path = f"{base_name}_edge.png"
    save_image(output_path, final_image)
    print(f"Edge detected image saved as {output_path}")

if __name__ == "__main__":
    image_path = './sample.png'
    canny_edge_detection(image_path)