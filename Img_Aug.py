import numpy as np 
import random
import math
import os
from PIL import Image, ImageEnhance
import cv2

# return a Image obj and a image numpy array, did not used in this pipeline actually
# It implements the conversion from Image obj to numpy array
def readImg(filename):
	img_obj = Image.open(filename)
	rgb_img = np.array(img_obj.convert("RGB"))
	img_array = rgb_img[:,:,::-1]
	return img_obj, img_array



def color_transfer_constructor(lMeanTar = 125.0491, lStdTar = 44.1218, \
							   aMeanTar = 160.7643, aStdTar = 7.5914, \
							   bMeanTar = 109.7114, bStdTar = 6.0879, \
							   l_range = (0.9, 1.1), a_range = (0.95, 1.05), b_range = (0.95, 1.05), constant_range = (-0.5, 0.5)):
	"""
	Transfers the color distribution from the source to the target
	image using the mean and standard deviations of the L*a*b*
	color space. Then apply a random modification on the transfered image.
	Parameters:
	-------
	source: NumPy array
		OpenCV image in BGR color space (the source image)
	lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar: Scalars
		Means and Stds Calculated from the target image
	l_range, a_range, b_range, constant_range: tuple
	    Range of the random modification
	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (uint8)
	"""
	def color_transfer(source):
		binary = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
		binary[binary<=215] = 0
		binary[binary>215] = 1

		# convert the images from the RGB to L*ab* color space, being
		# sure to utilizing the floating point data type (note: OpenCV
		# expects floats to be 32-bit, so use that instead of 64-bit)
		source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")


		l = source[:,:,0]
		a = source[:,:,1]
		b = source[:,:,2]
		(lMeanSrc, lStdSrc) = (l[np.where(binary==0)].mean(), l[np.where(binary==0)].std())
		(aMeanSrc, aStdSrc) = (a[np.where(binary==0)].mean(), a[np.where(binary==0)].std())
		(bMeanSrc, bStdSrc) = (b[np.where(binary==0)].mean(), b[np.where(binary==0)].std())


		l = (l - lMeanSrc) / lStdSrc * lStdTar + lMeanTar
		a = (a - aMeanSrc) / aStdSrc * aStdTar + aMeanTar
		b = (b - bMeanSrc) / bStdSrc * bStdTar + bMeanTar


		# randomly modify the lab space to do color augmentation
		l = random.uniform(l_range[0], l_range[1]) * l + random.uniform(constant_range[0], constant_range[1])     # Adjust Luminance
		a = random.uniform(a_range[0], a_range[1]) * a + random.uniform(constant_range[0], constant_range[1])   # Adjust color
		b = random.uniform(b_range[0], b_range[1]) * b + random.uniform(constant_range[0], constant_range[1])   # Adjust color

		# clip the pixel intensities to [0, 255] if they fall outside
		# this range
		l = np.clip(l, 0, 255)
		a = np.clip(a, 0, 255)
		b = np.clip(b, 0, 255)

		# merge the channels together and convert back to the RGB color
		# space, being sure to utilize the 8-bit unsigned integer data
		# type
		transfer = cv2.merge([l, a, b])
		transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)
		
		# return the color transferred image
		return transfer
	return color_transfer 


def contrast_enhance_constructor(level_range=(0.7, 1.3)):
	"""
	Perform contrast enhancement with random intensity level.
	Parameters:
	-----------
	img: Numpy array
		OpenCV image in BGR color space
	level_range: Tuple
		Range of the random contrast enhancement intensity
	Returns:
	-----------
	img_out_array: Numpy array
		OpenCV image in BGR color space
	"""
	def contrast_enhance(img):
		# Input Arg: img is a numpy array


		# Convert the numpy array image to Image obj for image enhancement
		img_obj = Image.fromarray(img[:,:,::-1], 'RGB')  # numpy array to Image obj, note the bgr to rgb

		# Contrast modification
		factor = random.uniform(level_range[0], level_range[1])
		enhancer = ImageEnhance.Contrast(img_obj)
		img_out = enhancer.enhance(factor)   # get Image obj
		# return img_out
		img_out_array = np.array(img_out.convert("RGB"))  # convert Image obj to numpy array
		img_out_array = img_out_array[:,:,::-1]
		return img_out_array
	return contrast_enhance

def rotation_mirror(img):
	"""
	Perform random rotation and mirroring. 
	Parameters: 
	------------
	img: Numpy array
		OpenCV image in BGR color space
	Returns:
	------------
	img_rot_mir: Numpy array
		OpenCV image in BGR color space
	"""
	R_state = random.uniform(0,4)
	M_state = random.uniform(0,4)
	# Apply rotation
	k = math.ceil(R_state)
	img_rot = np.rot90(img, k)
	if M_state<1:
		img_rot_mir = np.fliplr(img_rot) # Apply horizontal flip
	elif M_state<2:
		img_rot_mir = np.flipud(img_rot) # Apply vertical flip
	elif M_state<3:
		img_rot_mir = np.fliplr(img_rot) # Apply horizontal and vertical flip
		img_rot_mir = np.flipud(img_rot_mir)
	else:
		img_rot_mir = img_rot             # do nothing
	return img_rot_mir

def blur_constructor(sigma_range = (0.0001, 1.3)):
	"""
	Perform Gaussian blur with random blurring intensity.
	Parameters:
	-------------
	img: Numpy array
		OpenCV image in BGR color space
	sigma_range: Tuple
		range of the random blur intensity
	Returns:
	-------------
	img_blur: Numpy array
		OpenCV image in BGR color space
	"""
	def blur(img):
		sigma = random.uniform(sigma_range[0], sigma_range[1]) # extent of the blurrinng
		img_blur = cv2.GaussianBlur(img, (5,5), sigma)

		return img_blur
	return blur
def addGaussianNoise_constructor(noise_range = (0.001, 0.01)):
	"""
	Add Gaussian Noise to images, with random noise variation.
	Parameters:
	--------------
	img: Numpy array
		OpenCV image in BGR color space
	noise_range: Tuple
		Range of the random noise variation
	"""
	def addGaussianNoise(img):
		img = img.astype('float64')
		img *= 1./255    # Normalize first. Will recover later
		row, col, ch = img.shape
		sigma = random.uniform(noise_range[0], noise_range[1])
		sigma = sigma**0.5
		noise = np.random.normal(0, sigma, (row, col, ch))
		noise = noise.reshape(row,col,ch)
		img_noise = noise+img
		img_noise *= 255
		return img_noise
	return addGaussianNoise

def getStdMean(image):
	"""
	Parameters:
	-------
	image: NumPy array
		OpenCV image in BGR color space
	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""

	binary = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	binary[binary<=215] = 0
	binary[binary>215] = 1

	# compute the mean and standard deviation of each channel
	# (l, a, b) = cv2.split(image)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
	l = image[:,:,0][np.where(binary == 0)]
	a = image[:,:,1][np.where(binary == 0)]
	b = image[:,:,2][np.where(binary == 0)]




	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())

	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)















# source = cv2.imread('source.jpg')
# for i in range(9):
# 	noise = addGaussianNoise(source)
# 	transfer = blur_noise(noise)
# 	newfilename = 'source_removenoise' + str(i+1) + '.jpg'
# 	cv2.imwrite(newfilename, transfer)
# print 'Done.'

#############################
# For color_transfer and contrast_enhance, take 0.6-0.7 hour to finish on 3479 images.
# The entire process takes around 1h (roughly) to finish on 3479 images.
#############################

# input_path = './output/PAPILLARY_LEPIDIC_after_screen/'
# output_path = './output/py_PAPILLARY_LEPIDIC_Aug/'
# print 'Getting file names ...'
# # filenames = os.listdir(input_path)
# filenames = [i for i in os.listdir(input_path) if i.endswith('.jpg')]
# # filenames = [i for i in os.listdir(input_path) if i.startswith('LAD-GP-0003_PAPILLARY')]
# print '{0} images found.'.format(len(filenames))

# target = cv2.imread('target.jpg')    # tar is a Image obj, not a numpy array, target is a numpy array
# (lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = getStdMean(target)
# cnt = 1
# for filename in filenames:
# 	source = cv2.imread(input_path + filename)
# 	filename = filename.strip('.jpg')
# 	newfilename = filename + '_0' + '.jpg'
# 	# Do nothing to original image
# 	cv2.imwrite(output_path + newfilename, source)
# 	for i in range(9):
# 		# T:color_transform C: contrast R:rotation and mirroring G: Gaussian filter and noise
# 		source_T = color_transfer(source, lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) # Color Transform
# 		source_TC = contrast_enhance(source_T)
# 		source_TCR = rotation_mirror(source_TC)
# 		source_TCRG = blur_noise(source_TCR)


# 		newfilename = filename + '_' + str(i+1) + '.jpg'
# 		# img_out.save(output_path + newfilename)
# 		cv2.imwrite(output_path + newfilename, source_TCRG)
# 	cnt += 1
# 	if (cnt % 100 ==0):
# 		print ('Now processed {0} images.'.format(str(cnt)))
# print 'Done.'






