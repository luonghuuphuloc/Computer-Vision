import numpy as np
import cv2

#Read template image
template = cv2.imread('template.png',0)
template = cv2.medianBlur(template,3)

# Find canny of template image
edges = cv2.Canny(template,7,46)
kernel = np.ones((7,7),np.uint8)
#cv2.imshow('before close',edges)
edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
#cv2.imshow('after close',edges)
# Find contours of object in image
_, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(template,contours,0,(0,255,0),2)
res = np.hstack((edges,template))
cv2.imshow('res',res)
cv2.imwrite('res.png',res)
# Choose biggest contour as object's contour
cnt = contours[0]


# Read image which has objects that we need to find
img = cv2.imread('image.png',1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Process image to make better result
gray = cv2.GaussianBlur(gray,(3,3),0)
binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,77,6)
kernel = np.ones((3,3),np.uint8)
cv2.imshow('after close',binary)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
cv2.imshow('binary',binary)

# Find contours
_, contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find objects
for i in range(len(contours)):
    if cv2.contourArea(contours[i]) > 1000 :

        # Calculate matchShape value
        ret = cv2.matchShapes(contours[i],cnt,1,0.0)
        if ret < 1: #Threshold value

            # Draw circle at center 
            M = cv2.moments(contours[i])
            cx = int(M['m10']/M['m00'])
            cy = int(M['m01']/M['m00'])
            cv2.circle(img,(cx,cy),5,[0,0,255],-1)

            # Draw rectangle around object
            rect = cv2.minAreaRect(contours[i])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,255,0),2)
            cv2.imshow('img',img)  

cv2.imwrite('img.png',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

