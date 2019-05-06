import numpy as np
import cv2
img=cv2.imread('C:/Users/jSpecter/Documents/Research/Datasets/extended-cohn-kanade-images/cohn-kanade-images/S055/005/S055_005_00000043.png')
#img=cv2.imread('C:/Users/jSpecter/Documents/Research/Datasets/extended-cohn-kanade-images/cohn-kanade-images/S035/002/S035_002_00000008.png')

w=256
h=256
dim=(w,h)

img=cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

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
	#cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
	roi_gray=img[y:y+h,x:x+w]

cv2.imshow('image',roi_gray)
cv2.waitKey(1000)

#Scaling the face

w=256
h=256
dim=(w,h)

resized=cv2.resize(roi_gray, dim, interpolation = cv2.INTER_AREA)
cv2.imshow('image',resized)
cv2.waitKey(1000)

#Histogram Equalization
resized = cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
cv2.imshow('image',resized)
cv2.waitKey(1000)

equ=cv2.equalizeHist(resized)
cv2.imshow('image',equ)
cv2.waitKey(1000)

equz=cv2.equalizeHist(resized)
cv2.imshow('image',equ)
cv2.waitKey(1000)
#Eyes

eyes_x=eyes_y=eyes_h=eyes_w=[0,0]
eyes_c=[[0,0],[0,0]]
counter=0
eye_cascade=cv2.CascadeClassifier('haarcascade_eye.xml')
eyes=eye_cascade.detectMultiScale(equ,3)
for (ex,ey,ew,eh) in eyes:
	eyes_x[counter]=ex
	eyes_y[counter]=ey
	eyes_w[counter]=ew
	eyes_h[counter]=eh
	roi_gray1=equ[ey:ey+eh,ex:ex+ew]
	eyes_c[counter]=[ex+int(ew/2),ey+int(eh/2)]
	print("Cords:"+str(ex)+":"+str(ey)+":"+str(eh)+":"+str(ew))
	#cv2.rectangle(equ, (ex, ey), (ex+ew,ey+eh), (0,255,0), 2)
	cv2.circle(equ,(eyes_c[counter][0],eyes_c[counter][1]),3,(255,255,255), -1)
	cv2.imshow('roi',roi_gray1)
	cv2.waitKey(1000)
	counter=counter+1
cv2.imshow('image',equ)
cv2.waitKey(1000)


#Nose
nose_x=nose_y=nose_w=nose_h=0
nose_cx=nose_cy=0
nose_cascade=cv2.CascadeClassifier('haarcascade_nose.xml')
nose=nose_cascade.detectMultiScale(equ,3)
for (x,y,w,h) in nose:
	nose_x=x
	nose_y=y
	nose_w=w
	nose_h=h
	roi_gray2=equ[y:y+h,x:x+w]
	print("Cords:"+str(x)+":"+str(y)+":"+str(h)+":"+str(w))
	nose_cx=x+int(w/2)
	nose_cy=y+int(h/2)
	cv2.circle(equ,(nose_cx,nose_cy),3,(255,255,255), -1)
cv2.imshow('image',roi_gray2)
cv2.waitKey(1000)
cv2.imshow('image',equ)
cv2.waitKey(1000)

center=0
for i in range(len(eyes_x)):
	center=center+eyes_c[i][0]
center=int(center/2)
cv2.rectangle(equ, (center-14, (eyes_c[0][1]-14)-28), (center+14, (eyes_c[0][1]+14)-28), (0,255,0), 2)
cv2.rectangle(equ, (center-14, eyes_c[0][1]-14), (center+14, eyes_c[0][1]+14), (0,255,0), 2)
cv2.rectangle(equ, (eyes_c[0][0]-14, eyes_c[0][1]+5), (eyes_c[0][0]+14, eyes_c[0][1]+33), (0,255,0), 2)
cv2.rectangle(equ, (eyes_c[1][0]-14, eyes_c[1][1]+5), (eyes_c[1][0]+14, eyes_c[1][1]+33), (0,255,0), 2)
cv2.rectangle(equ, (nose_x-28, nose_cy), (nose_x, nose_cy+28), (0,255,0), 2)
cv2.rectangle(equ, (nose_x+nose_w, nose_cy), (nose_x+nose_w+28, nose_cy+28), (0,255,0), 2)
cv2.rectangle(equ, (nose_x-56, nose_cy), (nose_x-28, nose_cy+28), (0,255,0), 2)
cv2.rectangle(equ, (nose_x+nose_w+28, nose_cy), (nose_x+nose_w+56, nose_cy+28), (0,255,0), 2)
cv2.rectangle(equ, (nose_x-56, nose_cy+28), (nose_x-28, nose_cy+56), (0,255,0), 2)
cv2.rectangle(equ, (nose_x+nose_w+28, nose_cy+28), (nose_x+nose_w+56, nose_cy+56), (0,255,0), 2)
center_en1=[int((eyes_c[0][0]+nose_cx)/2),int((eyes_c[0][1]+nose_cy)/2)]
center_en2=[int((eyes_c[1][0]+nose_cx)/2),int((eyes_c[1][1]+nose_cy)/2)]
cv2.rectangle(equ, (center_en1[0]-14,center_en1[1]-14), (center_en1[0]+14,center_en1[1]+14), (0,255,0), 2)
cv2.rectangle(equ, (center_en2[0]-14,center_en2[1]-14), (center_en2[0]+14,center_en2[1]+14), (0,255,0), 2)
cv2.imshow('image',equ)
cv2.waitKey(5000)

#Eyebrows
for i in range(len(eyes_x)):
	eyebrows=equz[eyes_y[i]-int(eyes_h[i]/2):eyes_y[i]+int(eyes_h[i]/2),eyes_x[i]:eyes_x[i]+(eyes_w[i])]
	cv2.imshow('image',eyebrows)
	cv2.waitKey(1000)

th2 = cv2.adaptiveThreshold(eyebrows,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imshow('image',th2)
cv2.waitKey(5000)

sobely = cv2.Sobel(eyebrows,cv2.CV_64F,0,1,ksize=3)
cv2.imshow('image',sobely)
cv2.waitKey(5000)

ret,thresh=cv2.threshold(sobely,127,255,cv2.THRESH_BINARY)
cv2.imshow('image',thresh)
cv2.waitKey(5000)

#Smile

face_cascade=cv2.CascadeClassifier('haarcascade_smile.xml')
faces=face_cascade.detectMultiScale(equ,4)
for (x,y,w,h) in faces:
	roi_gray3=equz[y:y+h,x:x+w]
	print("Cords:"+str(x)+":"+str(y)+":"+str(h)+":"+str(w))
cv2.imshow('image',roi_gray3)
cv2.waitKey(1000)

blur = cv2.GaussianBlur(roi_gray3,(3,3),0)
cv2.imshow('image',blur)
cv2.waitKey(1000)

sobely = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)
cv2.imshow('image',sobely)
cv2.waitKey(5000)

ret,thresh=cv2.threshold(sobely,127,255,cv2.THRESH_BINARY)
cv2.imshow('image',thresh)
cv2.waitKey(5000)

'''for i in range(smile_x,smile_x+smile_w):
	for j in range(smile_y,smile_y+smile_h):
		if(thresh[i][j]==255):
'''

th2 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
cv2.imshow('image',th2)
cv2.waitKey(5000)

#Salient Patches

