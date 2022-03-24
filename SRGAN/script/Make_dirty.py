import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

train_dir = r'C:\Users\user\Boulot'
it = 0
for elem in os.listdir(train_dir + "/datamatrix"):
    img = cv2.imread(train_dir + "/datamatrix/" + elem)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (30, 30))

    new_img = []
    for i in img:
        for elem in i:
            if elem == 255:
                elem = elem - 100
                new_img.append(elem)

            else:
                elem = elem + 50
                new_img.append(elem)

    new_img = np.reshape(new_img, (30, 30))
    new_img = np.array(new_img, dtype=np.uint8)
    plt.imshow(new_img)

    cv2.imwrite(train_dir + "/dirty_datama/" + img + '.jpg', new_img)
