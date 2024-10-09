# Function to generate a zigzag pattern from a square matrix
def generate_zigzag_pattern(input_matrix):
    dimension = len(input_matrix)
    zigzag_result = []

    # Iterate over the upper triangle of the matrix
    for i in range(dimension):
        for j in range(i + 1):
            if i % 2 == 0:
                zigzag_result.append(input_matrix[j][i - j])
            else:
                zigzag_result.append(input_matrix[i - j][j])

    # Iterate over the lower triangle of the matrix
    for i in range(1, dimension):
        for j in range(dimension - i):
            if (i + j) % 2 == 0:
                zigzag_result.append(input_matrix[dimension - 1 - j][i + j])
            else:
                zigzag_result.append(input_matrix[i + j][dimension - 1 - j])

    return zigzag_result

# Function to perform Run-Length Encoding (RLE) on a list of symbols
def run_length_encoding(data):
    encoded_data = []
    count = 1

    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded_data.extend([data[i - 1], count])
            count = 1

    encoded_data.extend([data[-1], count])
    return encoded_data
