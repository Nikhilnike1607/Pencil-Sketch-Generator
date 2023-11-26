#!/usr/bin/env python
# coding: utf-8

# **Image to Pencil sketch**

# In[89]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2


# In[90]:


input_img = 'Dhoni.jpg'
org_img = cv2.imread(input_img)


# In[91]:


original_img_rgb = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
plt.imshow(original_img_rgb)
plt.axis('off')
plt.title('Original Image')
plt.show()


# In[92]:


gray_image = cv2.cvtColor(org_img, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image)
plt.axis('off')
plt.title('Grayscale Image')
plt.show()


# In[93]:


inverted_gray_image = cv2.bitwise_not(gray_image)
plt.imshow(inverted_gray_image)
plt.axis('off')
plt.title('Inverted grey Image')
plt.show()


# In[94]:


blurred_img = cv2.GaussianBlur(inverted_gray_image, (111,111),0)
plt.imshow(blurred_image)
plt.axis('off')
plt.title('Blurred Image')
plt.show()


# In[83]:


inverted_blurred_image = cv2.bitwise_not(blurred_img)
plt.imshow(inverted_blurred_image)
plt.axis('off')
plt.title('blurred Image')
plt.show()


# In[107]:


pencil_sketch = cv2.divide(gray_image, inverted_blurred_image, scale =220)


# In[108]:


plt.imshow(pencil_sketch)
plt.axis('off')
plt.show()


# In[109]:


pencil_sketch_rgb = cv2.cvtColor(pencil_sketch, cv2.COLOR_GRAY2RGB)


# In[110]:


plt.imshow(pencil_sketch_rgb)
plt.axis('off')
plt.title('Pencil sketch')
plt.show()


# In[ ]:




