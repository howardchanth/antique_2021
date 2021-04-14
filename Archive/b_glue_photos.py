import os
from PIL import Image
import numpy as np


# 5000 images storing the surrounding
surr_image_dir = os.path.join("./Data", "A_5000")
# Images storing top and down of the antique
top_bottom_image_dir = os.path.join("./Data", "top_and_bottom")

surr_images_paths = [os.path.join(surr_image_dir, file_name) for file_name in os.listdir(surr_image_dir)]
top_bottom_image_paths = [os.path.join(top_bottom_image_dir, file_name)
                          for file_name in  os.listdir(top_bottom_image_dir)]

# Step 1: Load Images into numpy array
imgs = []

for i in range(len(surr_images_paths)):
    path = surr_images_paths[i]

    if i % 8 == 0:
        img = Image.open(path)
        imgs.append(np.asarray(img))
        img.close()


# Step 2: Load base image
# display(image)
base = imgs[0]


# Step 3: Glue the images into one flattened feature map
# Using the middle 4 pixel vectors for the feature map
# O(n^2) algorithm to glue the images
window = None
indx = 0
merge = base[105:2010, 578:1078]

while indx < len(imgs):
    # Get the next window to merge
    # Get the middle of the base vase as the starting window
    if window is None:
        window = base[105:2010, 1078:1082]
        rightmost = window[:, -1, :]
    # Compute difference of the rightmost vector of the base window and leftmost vectors of windows
    # of all the vases. Take the window having the smallest difference.
    else:
        distance = []
        for img in imgs[indx:]:
            # Crop the image and convert it to numpy
            data = img[105:2010, 1078:1082]
            # Get the left most vector
            leftmost = data[:, 0, :]
            distance.append(np.sum((rightmost - leftmost) * (rightmost - leftmost)))

        indx += np.argmin(distance)
        window = imgs[indx][105:2010, 1078:1082]
        # Update rightmost vector
        rightmost = window[:, -1, :]

    merge = np.concatenate((merge, window), axis=1)
    print(f"Pasted image {indx}")
    indx += 1
# Paste the right side of the vase
merge = np.concatenate((merge, imgs[indx-1][105:2010, 1082:1560]), axis=1)

Image.fromarray(merge).show()
Image.fromarray(merge).save("./glued_surr.jpg")

# Combine with top and bottom images to obtain the feature map
top_bottom = [Image.open(path) for path in top_bottom_image_paths]
top_bottom = [np.asarray(img.resize((1905, 1905))) for img in top_bottom]
top_bottom = np.concatenate(top_bottom, axis=1)
glued_all = np.concatenate((merge, top_bottom), axis=1)

Image.fromarray(glued_all).show()
Image.fromarray(glued_all).save("./glued_all.jpg")
