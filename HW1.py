import numpy as np
import scipy.signal as scisig
import scipy.ndimage.filters as scifilt

OriginalImage = np.array([[4,1,6,1,3],
                          [3,2,7,7,2],
                          [2,5,7,3,7],
                          [1,4,7,1,3],
                          [0,1,6,4,4]]).astype(dtype='Float32')

mean_filter = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]]) / 9

print("Original matrix:")
print(OriginalImage)

OriginalWithMeanFilter = scisig.convolve2d(OriginalImage,mean_filter)

print("\nApplying mean filter:")
print(OriginalWithMeanFilter)

# default size for this function is 3X3
OriginalWithMedianFilter = scisig.medfilt2d(OriginalImage)

print("\nApplying median filter:")
print(OriginalWithMedianFilter)

# Sigma chosen randomly.
OriginalWithGaussianFilter = scifilt.gaussian_filter(OriginalImage,sigma=0.5)

print("\nApplying gaussian filter:")
print(OriginalWithGaussianFilter)


