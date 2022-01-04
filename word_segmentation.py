import cv2
import numpy as np
import matplotlib.pyplot as plt

raw = cv2.imread("input/neat_one_line.jpg")
gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
(thresh, img) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

horizontal_histogram = img.shape[1] - np.sum(img, axis=1, keepdims=True) / 255
vertical_histogram = img.shape[0] - np.sum(img, axis=0, keepdims=True) / 255
plt.plot([[el] for el in vertical_histogram[0]])
plt.fill([[el] for el in vertical_histogram[0]])
vertical_histogram = np.array(vertical_histogram).flatten()
print(vertical_histogram)

it = np.nditer(vertical_histogram, flags=['f_index'])
counter = 0
indices = []
temp = []
for x in it:
    if x < 5:
        counter += 1
        temp.append(it.index)
        if counter > 20:
            indices += temp
            temp = []
    else:
        temp = []
        counter = 0

for x in indices:
    cv2.line(img, (x, 0), (x, img.shape[0]), (0, 0, 0), 2)

cv2.imshow("grayscale", img)

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
