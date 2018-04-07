import cv2 as cv 
import numpy as np 

image=cv.imread('opencv-template-matching-python-tutorial.jpg')
template=cv.imread('opencv-template-for-matching.jpg')

h,w,a=template.shape

match=cv.matchTemplate(image,template,cv.TM_CCOEFF_NORMED)
threshold=0.8 #Percentage of matching

pos=np.where(match>=threshold)

for pt in zip(*pos[::-1]):
	cv.rectangle(image,pt,(pt[0]+w,pt[1]+h),(0,0,255),2)

h,w,_=image.shape
resize=cv.resize(image,(h/4,w/4))

cv.imshow('output',resize)
cv.waitKey(0)
cv.destroyAllWindows()