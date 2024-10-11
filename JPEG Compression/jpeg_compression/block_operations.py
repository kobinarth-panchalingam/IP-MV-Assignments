# Function to extract an 8x8 block from an image at position (x, y)
def block_8x8(image, x, y):
    block = []
    for i in range(8):
        row = []
        for j in range(8):
            row.append(image[y + i][x + j])
        block.append(row)
    return block

# Function to select a specific channel (Y, Cr, or Cb) from an 8x8 block
def select_channel(block, channel):
    block_channel = []
    index = 0 if channel == 'Y' else (1 if channel == 'Cr' else 2)
    for i in range(8):
        row = []
        for j in range(8):
            row.append(block[i][j][index])
        block_channel.append(row)
    return block_channel