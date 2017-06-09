# Based on the tutorial for using Gaussian pyramids from scikit-image.
# Link: http://scikit-image.org/docs/dev/auto_examples/transform/plot_pyramid.html
# I wrote my own function for computing the pyramids and display a comparison of the two.

import matplotlib.pyplot as plt
import skimage.filters as scifilt
import numpy as np
from scipy.misc import imresize
from skimage import data, color
from skimage.transform import pyramid_gaussian, pyramid_laplacian


def display_image(axis, pic, title):
    axis.imshow(pic)
    axis.axis('off')
    axis.set_title(title)


def display_image_gray(axis, pic, title):
    axis.imshow(pic, cmap='gray')
    axis.axis('off')
    axis.set_title(title)


def CreateGaussianAndLaplacianPyramid(pic, level):
    FilteredPic = scifilt.gaussian(pic, sigma=0.5, mode='nearest', multichannel=True)
    # mode='nearest' is the mode chosen by scikit-image pyramids filters.
    DownsampledPic = imresize(FilteredPic, 0.5)

    Gaussian = [pic]
    Laplacian = [color.rgb2gray(pic) - color.rgb2gray(FilteredPic)]

    if level > 1:
        TempGauss, TempLap = CreateGaussianAndLaplacianPyramid(DownsampledPic, level - 1)
        GaussianPyramid = Gaussian + TempGauss
        LaplacianPyramid = Laplacian + TempLap
    else:
        GaussianPyramid = Gaussian
        LaplacianPyramid = Laplacian

    return GaussianPyramid, LaplacianPyramid


image = data.astronaut()
rows, cols, dim = image.shape

# original code from example
GaussianPyramid = tuple(pyramid_gaussian(image, downscale=2))
LaplacianPyramid = tuple(pyramid_laplacian(image, downscale=2))

composite_image_gauss = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
composite_image_lapla = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)

composite_image_gauss[:rows, :cols, :] = GaussianPyramid[0]
composite_image_lapla[:rows, :cols, :] = LaplacianPyramid[0]

i_row = 0
for p in GaussianPyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image_gauss[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

i_row = 0
for p in LaplacianPyramid[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image_lapla[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig1, ax1 = plt.subplots(1,2)
display_image(ax1[0], composite_image_gauss, 'scikit-image\'s pyramid_gaussian')
display_image_gray(ax1[1], color.rgb2gray(composite_image_lapla), 'scikit-image\'s pyramid_laplacian')

# My code
Gaussian, Laplacian = CreateGaussianAndLaplacianPyramid(image, 8)

composite_image2_gauss = np.zeros((rows, cols + cols // 2, 3), dtype=np.double)
composite_image2_lapla = np.zeros((rows, cols + cols // 2), dtype=np.double)
composite_image2_gauss[:rows, :cols, :] = Gaussian[0]
composite_image2_lapla[:rows, :cols] = Laplacian[0]

i_row = 0
for p in Gaussian[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image2_gauss[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

i_row = 0

for p in Laplacian[1:]:
    n_rows, n_cols = p.shape[:2]
    composite_image2_lapla[i_row:i_row + n_rows, cols:cols + n_cols] = p
    i_row += n_rows

fig2, ax2 = plt.subplots(1,2)
display_image(ax2[0], composite_image2_gauss.astype(np.float64) / 255, 'My CreateGaussianAndLaplacianPyramid: Gauss')
display_image_gray(ax2[1], composite_image2_lapla.astype(np.float64) / 255,
                   'My CreateGaussianAndLaplacianPyramid: Laplace')

plt.show()
