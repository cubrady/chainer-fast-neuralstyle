#-*- coding:utf-8 -*-

import os
from config import *
from time import gmtime, localtime, strftime

def writeToFileLog(msg, logfile = LOG_FILE_GENERAL):
    log = os.path.join(LOG_FODLER, logfile)
    with open(log, 'a') as f:
        opt = "\r\n[%s]%s" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), msg)
        f.write(opt)
        print(opt)

def log(msg):
    print ("[%s] %s" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), msg))
    #print ("[%s] %s" % (, msg))
