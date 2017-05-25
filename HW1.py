import numpy as np
import scipy.signal as scisig

original_image = np.array([[4,1,6,1,3],
                           [3,2,7,7,2],
                           [2,5,7,3,7],
                           [1,4,7,1,3],
                           [0,1,6,4,4]]).astype(dtype='Float32')

mean_filter = np.array([[1,1,1],
                        [1,1,1],
                        [1,1,1]]) / 9

print("Original matrix:")
print(original_image)

OriginalWithMeanFilter = scisig.convolve2d(original_image,mean_filter)

print("\nConvolution with mean filter:")
print(OriginalWithMeanFilter)

# default size for this function is 3X3
OriginalWithMedianFilter = scisig.medfilt2d(original_image)

print("\nApplying median filter:")
print(OriginalWithMedianFilter)
