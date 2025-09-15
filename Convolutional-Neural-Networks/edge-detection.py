"""
This file contains from scratch implementation of convolution operation in numpy for edge detection

Step1: we require input and kernel matrices
considering image: (weight, height, depth) => (6, 6, 3) 
and we go ahead to convolution operation!!

"""
import numpy as np

#filter size = 3*3 for horizontal edge detection
kernel = np.array([
    [1, 1, 1],
    [0, 0, 0],
    [-1,-1,-1]
])

image = np.array([
    [1, 2, 3, 0, 0, 0],
    [4, 5, 6, 0, 0, 0],
    [7, 8, 9, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0]
])

def conv2d(img, kernel, padding, stride):

    #extract dimensions
    input_height, input_width = img.shape
    kernel_height, kernel_width = kernel.shape

    # apply padding
    if padding > 0:
        padded = np.pad(img, pad_width = padding, constant_values = 0)
    
    #calculate output dim
    output_w = (input_width - kernel_width + 2 * padding // stride) + 1
    output_h = (input_height - kernel_height + 2 * padding // stride) + 1

    output = np.zeros((output_h, output_w))

    for rows in range(input_height):
        for cols in range(input_width):
            x = rows * stride
            y = cols * stride

            region = padded[x : x + kernel_height, y: y + kernel_width]
            output[rows, cols] = np.sum(region * kernel)
    return output

print(conv2d(image, kernel, padding = 1, stride = 1))