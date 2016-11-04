#-*- coding:utf-8 -*-

import os, time
from config import *
from util import *

STRESS_TARGET_FILE = "tubingen.s720.jpg"

def __log(msg):
    writeToFileLog(msg, LOG_FILE_STRESS)

def __stressTest(*args):
    lock, queue, funcStressTarget = args
    path = os.path.join("sample_images", STRESS_TARGET_FILE)
    mode = 1
    model = "seurat"
    dicRet = funcStressTarget(path, mode, model, lock = lock)
    queue.put(dicRet[RET_TIME])

def stressTestThreadCount(count, funcStressTarget):
    import threading, thread, Queue
    lstThreads = []
    lock = thread.allocate_lock()
    queue = Queue.Queue()
    startTime = time.time()
    for i in xrange(1, count+1):
        t = threading.Thread(target = __stressTest, args = (lock, queue, funcStressTarget))
        t.start()
        lstThreads.append(t)

    exec_time_sum = 0.0
    exec_time_cnt = 0
    for t in lstThreads:
        t.join()
        exec_time_sum += queue.get()
        exec_time_cnt += 1
    totalSpend = time.time() - startTime

    __log("[Round %d] Avg spend time : %f, count : %d" % (count, totalSpend / float(exec_time_cnt), exec_time_cnt))

def stressTest(funcStressTarget):
    if not os.path.exists(LOG_FODLER):
        os.makedirs(LOG_FODLER)

    loop = 100
    __log("----------------------- START ----------------------- STRESS_TARGET_FILE:%s" % STRESS_TARGET_FILE)
    for i in xrange(1, loop):
        __log("\r\n\r\nStresst test Round %d >>>>>>>>>>>>>>>>>>" % i)
        stressTestThreadCount(i, funcStressTarget)
        __log("Stresst test Round %d Complete !!! " % i)
