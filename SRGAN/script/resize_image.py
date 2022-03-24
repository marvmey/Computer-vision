import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = 'random_data'

for img in os.listdir( train_dir+"/datamatrix"):
    img_array = cv2.imread( train_dir+"/datamatrix/" + img)

    img_array = cv2.resize(img_array, (120, 120))
    lr_img_array = cv2.resize(img_array, (30, 30))


    cv2.imwrite(train_dir + "/hr_datama/" + img, img_array)
    cv2.imwrite(train_dir + "/lr_datama/" + img, lr_img_array)


plt.imshow(img_array)
plt.axis('off')
plt.title("original")
plt.imshow(lr_img_array)
plt.axis('off')
plt.title("modified")




# train_dir = "data"
#
# for img in os.listdir(train_dir + "/original_images"):
#     img_array = cv2.imread(train_dir + "/original_images/" + img)
#
#     img_array = cv2.resize(img_array, (128, 128))
#     lr_img_array = cv2.resize(img_array, (32, 32))
#     cv2.imwrite(train_dir + "/hr_images/" + img, img_array)
#     cv2.imwrite(train_dir + "/lr_images/" + img, lr_img_array)