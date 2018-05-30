from Img_Aug import color_transfer_constructor, contrast_enhance_constructor, rotation_mirror, blur_constructor, addGaussianNoise_constructor
from Img_Aug import getStdMean
import cv2

target = cv2.imread('target.jpg')
source = cv2.imread('source.jpg')    # tar is a Image obj, not a numpy array, target is a numpy array
(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = getStdMean(target)


for i in range(9):

	color_transformer = color_transfer_constructor(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar, \
															l_range = (0.9, 1.1), a_range = (0.95, 1.05), b_range = (0.95, 1.05), constant_range = (-0.5, 0.5))
	img_out = color_transformer(source)
	newfilename = 'source_coloraug' + str(i+1) + '.jpg'
	cv2.imwrite(newfilename, img_out)

	contrast_enhancer = contrast_enhance_constructor(level_range=(0.7, 1.3))
	img_out = contrast_enhancer(source)
	newfilename = 'source_enhance' + str(i+1) + '.jpg'
	cv2.imwrite(newfilename, img_out)

	img_out = rotation_mirror(source)
	newfilename = 'source_rot_mirror' + str(i+1) + '.jpg'
	cv2.imwrite(newfilename, img_out)

	blurer = blur_constructor(sigma_range = (0.0001, 1.3))
	img_out = blurer(source)
	newfilename = 'source_blur' + str(i+1) + '.jpg'
	cv2.imwrite(newfilename, img_out)

	noise_adder = addGaussianNoise_constructor(noise_range = (0.001, 0.01))
	img_out = noise_adder(source)
	newfilename = 'source_noise' + str(i+1) + '.jpg'
	cv2.imwrite(newfilename, img_out)



print 'Done.'
