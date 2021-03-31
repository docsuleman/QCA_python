import cv2
import numpy as np

img =cv2.imread("images/disease.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

kernel=np.ones((3,3),np.float32)/9
#img=cv2.filter2D(img,-1,kernel)
#img = cv2.bilateralFilter(img,9,75,75)
#img=cv2.medianBlur(img,3)
img = cv2.GaussianBlur(img,(3,3),0)

sigma=0.3
median=np.median(img)
median=30
lower=int(max(0,(1.0-sigma)*median))
upper=int(min(255,(1.0+sigma)*median))

print(median, lower, upper)
auto_canny=cv2.Canny(img,lower,upper)

#auto_canny = cv2.medianBlur(auto_canny,3)

cv2.imshow("original",img)
cv2.imshow("result",auto_canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
