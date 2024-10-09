import heapq
from collections import defaultdict

# Define a HuffmanNode class for building the Huffman tree
class HuffmanNode:
    def __init__(self, freq, symbol, left=None, right=None):
        self.freq = freq
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

# Function to calculate symbol frequencies in the data
def calculate_symbol_frequencies(data):
    symbol_frequencies = defaultdict(int)
    for symbol in data:
        symbol_frequencies[symbol] += 1

    return symbol_frequencies

# Function to build a Huffman tree from symbol frequencies
def build_huffman_tree_from_frequencies(symbol_frequencies):
    priority_queue = [HuffmanNode(freq, symbol) for symbol, freq in symbol_frequencies.items()]
    heapq.heapify(priority_queue)

    while len(priority_queue) > 1:
        left = heapq.heappop(priority_queue)
        right = heapq.heappop(priority_queue)
        merged = HuffmanNode(left.freq + right.freq, None, left, right)
        heapq.heappush(priority_queue, merged)

    return heapq.heappop(priority_queue)

# Function to generate Huffman codes from a Huffman tree
def generate_huffman_codes_from_tree(huffman_tree):
    def assign_huffman_codes(node, code, huffman_codes):
        if node.symbol is not None:
            huffman_codes[node.symbol] = code
        else:
            assign_huffman_codes(node.left, code + "0", huffman_codes)
            assign_huffman_codes(node.right, code + "1", huffman_codes)

    huffman_codes = {}
    assign_huffman_codes(huffman_tree, "", huffman_codes)
    return huffman_codes

# Function to encode data using Huffman codes
def encode_data_with_huffman(data, huffman_codes):
    encoded_data = "".join(huffman_codes[symbol] for symbol in data)
    return encoded_data