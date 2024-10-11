import numpy as np

# Function to convert an RGB image to the YCrCb color space
def rgb_to_YCrCb(image, height, width):
    ycrcb_image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]
            y = 0.299 * r + 0.587 * g + 0.114 * b
            cr = int(128 + 0.5 * r - 0.418688 * g - 0.081312 * b)
            cb = int(128 - 0.168736 * r - 0.331264 * g + 0.5 * b)
            ycrcb_image[i, j] = [y, cr, cb]
    return ycrcb_image