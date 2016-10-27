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
RET_SETTING = "setting"

MAX_EDGE = 640

def resize(oriW, oriH):
	newW, newH = 0, 0
	if oriW > MAX_EDGE or oriH > MAX_EDGE:
		r = oriW / float(oriH)
		if oriW > oriH:
			newW = MAX_EDGE
			newH = int(newW / r)
		else:
			newH = MAX_EDGE
			newW = int(newH * r)
	return newW, newH

def resizeImage(inputPath):
	inputImage = Image.open(inputPath)
	print (inputImage.format, inputImage.size, inputImage.mode)
	oriW, oriH = inputImage.size

	newW, newH = resize(oriW, oriH)
	if newW * newH > 0:
		print("resize from %s to (%s, %d)" % (inputImage.size, newW, newH))
		inputImage = inputImage.resize((newW, newH), Image.BILINEAR)

	return inputImage, newW, newH

def generate(model, gpu, inputPath, median_filter, padding, out, mode = MODE_STATIC_ANIM_IMAGE):
	modelFastStyleNet = FastStyleNet()

	serializers.load_npz(model, modelFastStyleNet)
	if gpu >= 0:
	    cuda.get_device(gpu).use()
	    modelFastStyleNet.to_gpu()
	xp = np if gpu < 0 else cuda.cupy

	inputImage, oriW, oriH = resizeImage(inputPath)

	dicRet = {}
	dicRet[RET_MODE] = mode
	dicRet[RET_SETTING] = MAX_EDGE

	processTime = -1

	if mode == MODE_STATIC_ANIM_IMAGE:
		print("Mode is STATIC_ANIM_IMAGE")
		finalSize = None
		start = time.time()
		optImgList = []
		for i in xrange(0, DOWN_SCALE_COUNT):
			ratio = (DOWN_SCALE_COUNT - i) * DOWN_SCALE + DOWN_SCALE
			w, h = oriW - int(oriW * ratio), oriH - int(oriH * ratio)
			if not finalSize:
				finalSize = w, h
			#print ("Going to process %d x %d" % (w, h))
			t = time.time()
			nim = inputImage.resize( (w, h), Image.BILINEAR )
			print ("Resize done : %d x %d, spend %f sec" % (w, h, time.time() - t))

			ret = processImage(nim, xp, modelFastStyleNet, finalSize, i, padding, median_filter, out)
			print(ret)
			optImgList.append(ret)

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
	
	image = np.asarray(inputImage.convert('RGB'), dtype=np.float32).transpose(2, 0, 1)
	image = image.reshape((1,) + image.shape)
	if padding > 0:
		image = np.pad(image, [[0, 0], [0, 0], [padding, padding], [padding, padding]], 'symmetric')
	image = xp.asarray(image)
	x = Variable(image)

	y = model(x)
	result = cuda.to_cpu(y.data)

	result = np.uint8(result[0].transpose((1, 2, 0)))
	med = Image.fromarray(result)
	if median_filter > 0:
		med = med.filter(ImageFilter.MedianFilter(median_filter))
	
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
