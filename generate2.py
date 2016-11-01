#-*- coding:utf-8 -*- 

from __future__ import print_function
import numpy as np
import os, time
from PIL import Image, ImageFilter

import chainer
from chainer import cuda, Variable, serializers
from net import *

DOWN_SCALE = 0.02
DOWN_SCALE_COUNT = 5

MODE_STATIC_IMAGE = 1
MODE_STATIC_ANIM_IMAGE = 2

RET_TIME = "time"
RET_MODE = "mode"
RET_OPT_FILENAME = "file"
RET_OPT_FILENAME_LIST = "file_list"
RET_RESOLUTION = "resolution"
RET_MODEL = "model"

MAX_EDGE = 2048
MAX_EDGE_ANIM = 1024

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

	if mode == MODE_STATIC_ANIM_IMAGE:
		print("Mode is STATIC_ANIM_IMAGE")
		finalSize = None
		start = time.time()
		optImgList = []
		for i in xrange(0, DOWN_SCALE_COUNT):
			t = time.time()
			ratio = (DOWN_SCALE_COUNT - i) * DOWN_SCALE + DOWN_SCALE
			w, h = oriW - int(oriW * ratio), oriH - int(oriH * ratio)
			if not finalSize:
				finalSize = w, h
			
			nim = inputImage.resize( (w, h), Image.BILINEAR )
			optName = processImage(nim, xp, modelFastStyleNet, finalSize, i, padding, median_filter, out)
			optImgList.append(optName)

			print ("Round %d done : %d x %d, spend %f sec" % (i, w, h, time.time() - t))

		processTime = time.time() - start

		dicRet[RET_TIME] = processTime
		dicRet[RET_OPT_FILENAME_LIST] = optImgList

	elif mode == MODE_STATIC_IMAGE:
		print("Mode is STATIC_IMAGE")
		
		start = time.time()
		finalSize = oriW, oriH 
		optName = processImage(inputImage, xp, modelFastStyleNet, finalSize, 0, padding, median_filter, out)

		processTime = time.time() - start

		dicRet[RET_TIME] = processTime
		dicRet[RET_OPT_FILENAME] = optName

	else:
		print ("Err, unsupported parm : ", mode)

	print(">>>>>>>>>>>>>>>>> Process done, spend %d sec" % processTime)
	return dicRet

def processImage(inputImage, xp, model, targetSaveSize, idx, padding, median_filter, out):
	
	# Start of Code from generate.py

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

	# End of Code from generate.py

	name, ext = os.path.splitext(out)
	optName = "%s_%d%s" % (name , idx, ext)
	resizedMed = med.resize( targetSaveSize, Image.BILINEAR )
	resizedMed.save(optName)
	print (optName)
	return optName

if __name__=="__main__":
	model = "models/style.model"
	gpu = -1
	out = "out.jpg"
	inputPath = ""
	median_filter = 3
	padding = 50

	generate(model, gpu, inputPath, median_filterm, padding, out)
