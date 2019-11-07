import cv2

image = cv2.imread('example.png')
print(image.shape)

# notice
# bgr order
# y,x order
(b, g, r) = image[20, 100]  # accesses pixel at x=100, y=20
print(b, g, r)

cv2.imshow('Image', image)
cv2.waitKey(0)
