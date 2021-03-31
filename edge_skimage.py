
from scipy import ndimage as ndi
from skimage import feature,io
import matplotlib.pyplot as plt

from skimage.filters import meijering, sato, frangi, hessian
image=io.imread("images/disease.png",as_gray=True)
image = ndi.gaussian_filter(image, 2)


#very poor ridge_image = frangi(image)
#ridge_image = meijering(image,sigmas=[5],alpha=0.5)
#ridge_image = sato(image,sigmas=[7],black_ridges=True)
#ridge_image = hessian(image,black_ridges=True,mode='reflect',sigmas=[7])


edges_image = feature.canny(image,sigma=3,low_threshold=0.7,high_threshold=.99,use_quantiles=True)
#ridge_image = hessian(image,black_ridges=True,mode='reflect',sigmas=[7])


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original', fontsize=20)

ax[1].imshow(image, cmap='gray')
ax[1].set_title('Overlay', fontsize=20)

ax[1].imshow(edges_image, cmap='gray',alpha=0.5)

ax[2].imshow(edges_image, cmap='gray')
ax[2].set_title(r'Edge Only', fontsize=20)



for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
