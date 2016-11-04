#-*- coding:utf-8 -*-

from __future__ import print_function
import numpy as np
import os, time
from PIL import Image, ImageFilter
import images2gif
import chainer
from chainer import cuda, Variable, serializers
from net import *
from config import *

def getEdge(mode):
	return MAX_EDGE if mode == MODE_STATIC_IMAGE else MAX_EDGE_ANIM

def resize(oriW, oriH, mode):
	newW, newH = oriW, oriH
	edge = getEdge(mode)
	if oriW > edge or oriH > edge:
		r = oriW / float(oriH)
		if oriW > oriH:
			newW = edge
			newH = int(newW / r)
		else:
			newH = edge
			newW = int(newH * r)
	return newW, newH

def resizeImage(inputPath, mode):
	inputImage = Image.open(inputPath)
	print (inputImage.format, inputImage.size, inputImage.mode)
	oriW, oriH = inputImage.size

	newW, newH = resize(oriW, oriH, mode)
	if newW * newH > 0:
		print("resize from %s to (%s, %d)" % (inputImage.size, newW, newH))
		inputImage = inputImage.resize((newW, newH), Image.BILINEAR)

	return inputImage, newW, newH

def generate(model, gpu, inputPath, median_filter, padding, out, mode = MODE_STATIC_IMAGE):
	modelFastStyleNet = FastStyleNet()

	serializers.load_npz(model, modelFastStyleNet)
	if gpu >= 0:
	    cuda.get_device(gpu).use()
	    modelFastStyleNet.to_gpu()
	xp = np if gpu < 0 else cuda.cupy

	inputImage, oriW, oriH = resizeImage(inputPath, mode)

	dicRet = {}
	dicRet[RET_MODE] = mode
	dicRet[RET_RESOLUTION] = getEdge(mode)

	processTime = -1
	fileName, fileExt = os.path.splitext(out)

	if mode == MODE_STATIC_ANIM_IMAGE:
		print("Mode is STATIC_ANIM_IMAGE")
		finalSize = None
		start = time.time()
		optImgNameList = []
		imgList = []
		for i in xrange(0, DOWN_SCALE_COUNT):
			t = time.time()
			ratio = (DOWN_SCALE_COUNT - i) * DOWN_SCALE + DOWN_SCALE
			w, h = oriW - int(oriW * ratio), oriH - int(oriH * ratio)
			# Inorder to use ffmpeg to encode to video, the width must be even
			if w % 2 != 0:
				w += 1
			#print (i, w, h, ratio)
			if not finalSize:
				finalSize = w, h

			nim = inputImage.resize( (w, h), Image.BILINEAR )
			optName, resizedImg = processImage(nim, xp, modelFastStyleNet, finalSize, i, padding, median_filter, fileName, fileExt)
			optImgNameList.append(optName)
			imgList.append(resizedImg)

			print ("Round %d done : %d x %d, spend %f sec" % (i, w, h, time.time() - t))

		processTime = time.time() - start

		# Generate GIF
		giFileName = fileName + ".gif"
		images2gif.writeGif(giFileName, imgList, duration=0.5, dither=0)

		dicRet[RET_OPT_VIDEO] = genVideo(fileName)
		dicRet[RET_OPT_GIF] = giFileName
		dicRet[RET_TIME] = processTime
		dicRet[RET_OPT_FILENAME_LIST] = optImgNameList

	elif mode == MODE_STATIC_IMAGE:
		print("Mode is STATIC_IMAGE")

		start = time.time()
		finalSize = oriW, oriH
		optName, _ = processImage(inputImage, xp, modelFastStyleNet, finalSize, 0, padding, median_filter, fileName, fileExt)

		processTime = time.time() - start

		dicRet[RET_TIME] = processTime
		dicRet[RET_OPT_FILENAME] = optName

	else:
		print ("Err, unsupported parm : ", mode)

	print(">>>>>>>>>>>>>>>>> Process done, spend %d sec" % processTime)
	return dicRet

def genVideo(name):
	# -r 2 (2 frames per second)
	# ffmpeg -framerate 2 -i composition_%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p output.mp4
	optVideo = name + ".mp4"
	#cmd = "ffmpeg -f image2 -r 2 -i " + name + "_%d.jpg -vcodec mpeg4 -y " + optVideo
	cmd = "ffmpeg -framerate 2 -i " + name + "_%d.jpg -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p " + optVideo
	os.popen(cmd)
	#ret = os.popen(cmd).readlines()
	#print (ret)
	return optVideo

def processImage(inputImage, xp, model, targetSaveSize, idx, padding, median_filter, fileName, fileExt):

	# Start of Code from generate.py
	start = time.time()

	image = np.asarray(inputImage.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
	image = image.reshape((1,) + image.shape)
	if padding > 0:
		image = np.pad(image, [[0, 0], [0, 0], [padding, padding], [padding, padding]], 'symmetric')
	image = xp.asarray(image)
	x = Variable(image)

	y = model(x)
	result = cuda.to_cpu(y.data)

	if padding > 0:
		result = result[:, :, padding:-padding, padding:-padding]
	result = np.uint8(result[0].transpose((1, 2, 0)))
	med = Image.fromarray(result)
	if median_filter > 0:
		med = med.filter(ImageFilter.MedianFilter(median_filter))
	processTime = time.time() - start

	# End of Code from generate.py

	optName = "%s_%d%s" % (fileName , idx, fileExt)
	resizedMed = med.resize( targetSaveSize, Image.BILINEAR )
	resizedMed.save(optName)

	print (optName, processTime)
	return optName, resizedMed

if __name__=="__main__":
	model = "models/style.model"
	gpu = -1
	out = "out.jpg"
	inputPath = ""
	median_filter = 3
	padding = 50

	generate(model, gpu, inputPath, median_filterm, padding, out)
