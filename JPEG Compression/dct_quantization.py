import numpy as np
import math

# Function to calculate the 2D Discrete Cosine Transform (DCT) of an 8x8 block
def dct(block):
    N = 8
    dct_result = np.zeros((N, N), dtype=float)

    for u in range(N):
        for v in range(N):
            sum_val = 0

            for i in range(N):
                for j in range(N):
                    cu = 1 if u == 0 else math.sqrt(2)  # Adjust cu
                    cv = 1 if v == 0 else math.sqrt(2)  # Adjust cv
                    cos_u = math.cos((2 * i + 1) * u * math.pi / (2 * N))
                    cos_v = math.cos((2 * j + 1) * v * math.pi / (2 * N))

                    sum_val += cu * cv * block[i][j] * cos_u * cos_v

            dct_result[u][v] = 0.25 * sum_val

    return dct_result

# Standard quantization matrices for luminance and chrominance in JPEG compression
luminance_quantization_matrix = [
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
]

chrominance_quantization_matrix = [
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
]

# Function to apply quantization to an 8x8 block
def quantize_block(dct_block, quantization_matrix):
    quantized_block = [[0] * 8 for _ in range(8)]

    for i in range(8):
        for j in range(8):
            quantized_block[i][j] = round(dct_block[i][j] / quantization_matrix[i][j])

    return quantized_block