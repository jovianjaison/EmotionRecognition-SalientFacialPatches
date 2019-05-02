import numpy as np
import cv2
img=cv2.imread('jaffe/TM.HA2.181.tiff')

def aver(i,j):
	k=i
	l=j
	if not (k<0 or k>255 or j<0 or j>255):
		return img[k][l]/9
	else:
		return 0

cv2.imshow('image',img)
cv2.waitKey(1000)
#cv2.destroyAllWindows()

h,w,bpp = np.shape(img)
print("Height:"+str(h))
print("Width:"+str(w))
print("BPP:"+str(bpp))

#Low Pass Filtering

for i in range(0,h):
	for j in range(0,w):
		ans=0
		ans+=aver(i-1,j-1)
		ans+=aver(i,j-1)
		ans+=aver(i+1,j-1)
		ans+=aver(i-1,j)
		ans+=aver(i,j)
		ans+=aver(i+1,j)
		ans+=aver(i-1,j+1)
		ans+=aver(i,j+1)
		ans+=aver(i+1,j+1)
		img[i][j]=ans;

cv2.imshow('image',img)
cv2.waitKey(1000)

#HoG for face extraction

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces=face_cascade.detectMultiScale(img,1.3,5)
for (x,y,w,h) in faces:
	cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
	roi_gray=img[y:y+h,x:x+w]

cv2.imshow('image',img)
cv2.waitKey(1000)

cv2.imshow('image',roi_gray)
cv2.waitKey(1000)

#Scaling the face

w=256
h=256
dim=(w,h)

resized=cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('image',resized)
cv2.waitKey(1000)

