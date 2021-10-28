import cv2
import numpy as np 
import argparse
import imutils 
import utils  
import pytesseract



utils.initializeTrackbars()


img=cv2.imread('/home/sanskar/Pictures/icici.jpeg')
h,w=img.shape[:2]

def preProcessing(img):
	img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img_blur=cv2.GaussianBlur(img_gray,(5,5),1)
	while True:
		thresh=utils.valTrackbars()
		img_canny=cv2.Canny(img_blur,thresh[0],thresh[1])
		canny1=imutils.resize(img_canny,width=700)
		cv2.imshow('canny1' , canny1)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
	kernel=np.ones((5,5))
	img_dilation=cv2.dilate(img_canny,kernel,iterations=2)
	img_threshold=cv2.erode(img_dilation, kernel, iterations=1)
	return img_threshold

def getContours(img):
	biggest=np.array([])
	maxArea=0
	contours,_=cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for cnt in contours:
		area=cv2.contourArea(cnt)
		if area > 5000:
			cv2.drawContours(img_contour,cnt,-1,(0,255,0) ,3)
			# cv2.imshow('img_contour', img_contour)
			# cv2.waitKey(0)
			peri=cv2.arcLength(cnt,True)
				#print(peri)
			approx=cv2.approxPolyDP(cnt, 0.02*peri,True)
			if area > maxArea and len(approx)==4:
				biggest=approx
				maxArea=area
		cv2.drawContours(img_contour,biggest,-1,(0,255,0) ,3)
	return biggest

def reorder(myPoints):
	myPoints=myPoints.reshape((4,2))
	myPointsNew=np.zeros((4,1,2),np.int32)
	add=myPoints.sum(axis=1)
	print(add)
	myPointsNew[0]=myPoints[np.argmin(add)]
	myPointsNew[3]=myPoints[np.argmax(add)]
	diff=np.diff(myPoints,axis=1)
	myPointsNew[1]=myPoints[np.argmin(diff)]
	myPointsNew[2]=myPoints[np.argmax(diff)]
	return myPointsNew



# def getWarp(img,biggest):
# 	biggest=reorder(biggest)
# 	pts1=np.float32(biggest)
# 	pts2=np.float32([[0,0],[466,0],[0,350],[466,350]])
# 	matrix=cv2.getPerspectiveTransform(pts1,pts2)
# 	imgOutput=cv2.warpPerspective(img,matrix,(466,350))
# 	img_cropped=imgOutput
# 	img_cropped=imgOutput[10:imgOutput.shape[0]-10,10:imgOutput.shape[1]-10]
# 	img_cropped=cv2.resize(img_cropped,(350,466))
# 	return img_cropped


def getWarp(img,biggest):
	biggest=reorder(biggest)
	pts1=np.float32(biggest)
	pts2=np.float32([[0,0],[h,0],[0,w],[h,w]])
	matrix=cv2.getPerspectiveTransform(pts1,pts2)
	imgOutput=cv2.warpPerspective(img,matrix,(h,w))
	img_cropped=imgOutput
	img_cropped=imgOutput[10:imgOutput.shape[0]-10,10:imgOutput.shape[1]-10]
	img_cropped=cv2.resize(img_cropped,(w,h))
	return img_cropped
	

def sharpen(img):
	# kernel = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
	
	adaptive_threshold=cv2.adaptiveThreshold(img_warped_gray,255,1,1,7,2)
	# cv2.imshow('adaptive_threshold' , adaptive_threshold)
	adaptive_threshold=cv2.bitwise_not(adaptive_threshold)
	# cv2.imshow('bitwise_not' , adaptive_threshold)

	adaptive_threshold=cv2.medianBlur(adaptive_threshold,3)
	# cv2.imshow('blur' , adaptive_threshold)
	# cv2.waitKey(0)

	# img_sharpen = cv2.filter2D(img, -1, kernel)
	#adaptive_threshold=cv2.adaptiveThreshold(img_sharpen , 240 , cv2.ADAPTIVE_THRESH_GAUSSIAN_C , cv2.THRESH_BINARY , 199,5 )
	#thresh1 = cv2.adaptiveThreshold(img_sharpen , 255 , cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 199, 5)
	#cv2.imshow('adaptive_threshold',adaptive_threshold)
	#cv2.imshow('thresh1',thresh1)
	#cv2.waitKey(0)
	return adaptive_threshold


#img=imutils.resize(img,width=350)
#img=cv2.resize(img,None , fx=0.5, fy=0.5)
#print(img.shape)
#cv2.imshow('img' , img)


img_threshold=preProcessing(img)
img_contour=img.copy()
biggest=getContours(img_threshold)
img_warped=getWarp(img,biggest)
img_warped_gray=cv2.cvtColor(img_warped, cv2.COLOR_BGR2GRAY)
img_sharpen=sharpen(img_warped_gray)
print('img_threshold' , img_threshold.shape)
print('img_contour' , img_contour.shape)
print('biggest' , biggest.shape)
print('img_warped' , img_warped.shape)
print('img_warped_gray' , img_warped_gray.shape)
print('img_sharpen' , img_sharpen.shape)




# cv2.imshow('img',img)
# cv2.imshow('final' , img_contour)
# cv2.imshow('thresh' , img_threshold)
# cv2.imshow('warp' , img_warped)
# cv2.imshow('sharpen' , img_sharpen)
img_array=([img, img_threshold, img_contour] , [img_warped, img_warped_gray,img_sharpen])
# labels=[['original' ,'pre_processing' , 'biggest_contour_detected'], 
# 		['warp' , 'warped_gray' , 'sharpen']]

stackedImage=utils.stackImages(img_array,0.75)
stackedImage=imutils.resize(stackedImage, width=800)
cv2.imshow('res' , stackedImage)
# print(biggest)
# print(biggest.shape)


text=pytesseract.image_to_string(img_sharpen)
print(text)
cv2.waitKey(0)
cv2.destroyAllWindows()