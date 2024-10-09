import cv2
from color_conversion import rgb_to_YCrCb
from block_operations import block_8x8, select_channel
from dct_quantization import dct, quantize_block, luminance_quantization_matrix
from zigzag_rle import generate_zigzag_pattern, run_length_encoding
from huffman_encoding import calculate_symbol_frequencies, build_huffman_tree_from_frequencies, generate_huffman_codes_from_tree, encode_data_with_huffman

# Read the image
img = cv2.imread('input_image.jpeg')
height, width, _ = img.shape

# Convert to YCrCb
imgYCrCb = rgb_to_YCrCb(img, height, width)
cv2.imwrite('output_image.jpg', imgYCrCb)

# Extract 8x8 block and select Y channel
block = block_8x8(imgYCrCb, 100, 128)
selected = select_channel(block, "Y")
print("Selected channel Y\n")
for i in range(8):
    for j in range(8):
        print(selected[i][j], end="\t")
    print()

# Perform DCT and quantization
dct_result = dct(selected)
print("\nblock after dct\n")
dct = dct(selected)
for i in range(8):
  for j in range(8):
    print(round(dct[i][j]), end="\t")
  print()

quantized_block = quantize_block(dct_result, luminance_quantization_matrix)
print("\nQuantized block\n")
for i in range(8):
  for j in range(8):
    print(round(quantized_block[i][j]), end="\t")
  print()

# Generate zigzag pattern and perform RLE
ordered_list = generate_zigzag_pattern(quantized_block)
print("\nOrdered list\n")
print(ordered_list)

rle_result = run_length_encoding(ordered_list)
print("\nrun length encoding\n")
print(rle_result)

# Perform Huffman encoding
symbol_frequencies = calculate_symbol_frequencies(rle_result)
huffman_tree = build_huffman_tree_from_frequencies(symbol_frequencies)
huffman_codes = generate_huffman_codes_from_tree(huffman_tree)
encoded_data = encode_data_with_huffman(rle_result, huffman_codes)
print(encoded_data)
print(len(encoded_data))
