# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 21:11:03 2023
@author: nikhi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

def generate_pencil_sketch(input_img):
    org_img = cv2.imread(input_img)

    original_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
    plt.imshow(original_img_rgb)
    plt.axis('off')
    plt.title('Original Image')
    plt.show()

    gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.title('Black & White Image')
    plt.show()

    inverted_gray_image = cv2.bitwise_not(gray_image)
    plt.imshow(inverted_gray_image, cmap='gray')
    plt.axis('off')
    plt.title('Inverted Grey Image')
    plt.show()

    blurred_img = cv2.GaussianBlur(inverted_gray_image, (111, 111), 0)
    plt.imshow(blurred_img, cmap='gray')
    plt.axis('off')
    plt.title('Blurred Image')
    plt.show()

    inverted_blurred_image = cv2.bitwise_not(blurred_img)
    plt.imshow(inverted_blurred_image, cmap='gray')
    plt.axis('off')
    plt.title('Inverted Blurred Image')
    plt.show()

    pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale=220)
    plt.imshow(pencil_sketch, cmap='gray')
    plt.axis('off')
    plt.title('Pencil Sketch')
    plt.show()

    return inverted_blurred_image, pencil_sketch, blurred_img, inverted_gray_image, gray_image

# Example usage:
inverted_blurred, pencil_sketch, blurred, inverted_gray, gray = generate_pencil_sketch('Dhoni.jpg')
