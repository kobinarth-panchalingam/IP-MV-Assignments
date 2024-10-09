from PIL import Image
import numpy as np
import cv2

def nearest_neighbor_interpolation(input_image, new_width, new_height):
    original_width, original_height = input_image.size
    output_image = Image.new("RGB", (new_width, new_height))
    input_pixels = input_image.load()
    output_pixels = output_image.load()

    x_ratio = original_width / new_width
    y_ratio = original_height / new_height

    for x in range(new_width):
        for y in range(new_height):
            src_x = int(x * x_ratio)
            src_y = int(y * y_ratio)
            output_pixels[x, y] = input_pixels[src_x, src_y]

    return output_image

def bilinear_interpolation(input_image, new_width, new_height):
    original_width, original_height = input_image.size
    output_image = Image.new("RGB", (new_width, new_height))
    input_pixels = input_image.load()
    output_pixels = output_image.load()

    x_ratio = float(original_width - 1) / (new_width - 1)
    y_ratio = float(original_height - 1) / (new_height - 1)

    for x in range(new_width):
        for y in range(new_height):
            src_x = int(x * x_ratio)
            src_y = int(y * y_ratio)
            x_diff = (x * x_ratio) - src_x
            y_diff = (y * y_ratio) - src_y

            a = (1 - x_diff) * (1 - y_diff)
            b = x_diff * (1 - y_diff)
            c = (1 - x_diff) * y_diff
            d = x_diff * y_diff

            pixel1 = input_pixels[src_x, src_y]
            pixel2 = input_pixels[src_x + 1, src_y] if src_x < original_width - 1 else pixel1
            pixel3 = input_pixels[src_x, src_y + 1] if src_y < original_height - 1 else pixel1
            pixel4 = input_pixels[src_x + 1, src_y + 1] if src_x < original_width - 1 and src_y < original_height - 1 else pixel1

            r = int(pixel1[0] * a + pixel2[0] * b + pixel3[0] * c + pixel4[0] * d)
            g = int(pixel1[1] * a + pixel2[1] * b + pixel3[1] * c + pixel4[1] * d)
            b = int(pixel1[2] * a + pixel2[2] * b + pixel3[2] * c + pixel4[2] * d)

            output_pixels[x, y] = (r, g, b)

    return output_image

image_path = './input_image.jpeg'
input_image = cv2.imread(image_path)
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
input_image_pillow = Image.fromarray(input_image)

# new dimensions
new_width = 400
new_height = 300

# nearest neighbor interpolation
nearest_neighbor_result = nearest_neighbor_interpolation(input_image_pillow, new_width, new_height)
nearest_neighbor_result.save("nearest_neighbor_result.jpg")

# bilinear interpolation
bilinear_result = bilinear_interpolation(input_image_pillow, new_width, new_height)
bilinear_result.save("bilinear_result.jpg")
