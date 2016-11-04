#!flask/bin/python
import os, time
import threading
import random
import json
import ServerResponseDef
from time import gmtime, localtime, strftime
from flask import Flask, jsonify, abort, request, make_response, url_for
from flask import send_from_directory
from generate2 import generate, RET_MODEL
from config import *

app = Flask(__name__, static_url_path = "")

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/opt/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.root_path, OPT_FOLDER)
    return send_from_directory(directory=uploads, filename=filename)

@app.route('/style_list', methods=['GET'])
def get_model_list():
    if request.method == 'GET':
        prefix = os.path.join(OPT_FOLDER, THUMB_FOLDR, "%s_0.jpg")
        example = prefix % PRE_TRAINED_MODELS[0]
        dicRet = {"prefix":prefix, "style_list":PRE_TRAINED_MODELS, "example":example}
        return json.dumps(dicRet, ensure_ascii=False)

    dicRet = {"err":"invalid parameter"}
    return json.dumps(dicRet, ensure_ascii=False)

'''
http://www.mr-ping.com/post/Uvs0I8bMyEEMCgyd
'''
@app.route('/file', methods=['POST', 'GET'])
def update_file():
    """
    Api for getting file & string data from MApp
    Store data to local files
    """
    if request.method == 'POST':
        print ("\r\n\r\n ====================== upload task =====================")
        # Get file object from field of file
        f = request.files['userfile']
        path = os.path.join(FILE_UPLOAD_PATH, "%d_%s" % (time.time(), f.filename))
        f.save(path)
        # Get str object from field of text
        #s = request.form['name'] + "\r\n"
        print ("request:", request.form)
        mode = int(request.form['mode'])
        print ("mode : ", mode)
        model = request.form['model'] if request.form.has_key("model") else getModelName()
        if request.form.has_key("model"):
            print ("Use user defined model : %s" % model)
        dicRet = processImage(path, mode, model)
        dicRet["code"] = ServerResponseDef.SUCCESS

        print (dicRet)
        return json.dumps(dicRet, ensure_ascii=False)

    log('ERR_CMD_NOT_SUPPORT')
    dicRet = {"ret":"fail", "code":ServerResponseDef.ERR_CMD_NOT_SUPPORT}
    return json.dumps(dicRet, ensure_ascii=False)

def writeToFileLog(msg):
    log = os.path.join(FILE_UPLOAD_PATH, "log.txt")
    with open(log, 'a') as f:
        opt = "\r\n[%s]%s" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), msg)
        f.write(opt)
        print(opt)

def getModelName():
    #modelIdx = 0
    modelIdx = random.randint(0, len(PRE_TRAINED_MODELS) - 1)
    return PRE_TRAINED_MODELS[modelIdx]

def getModelPath(modelName):
    for path in MODEL_PATH_LIST:
        modelPath = os.path.join(path, modelName)
        if os.path.exists(modelPath):
            return modelPath
    writeToFileLog("[Err]Cannot find %s" % modelName)
    return ""

def processImage(path, mode, model, thumbMode = False):
    log ("processImage:%s" % path)

    modelName = "%s.model" % model
    modelPath = getModelPath(modelName)
    if modelPath:
        log("Use %s" % modelName)
    else:
        log("[ERROR] model %s is not exist" % modelPath)
        return processImage(path, mode, getModelName())

    gpu = 0 if ENABLE_GPU else -1
    log("Enable GPU:%s" % ENABLE_GPU)
    median_filter = 3
    padding = 50
    #storeFolder = "thumb_opt" if thumbMode else "%d_r%d" % (time.time(), random.randint(0, 100))
    storeFolder = "thumb_opt" if thumbMode else "%d" % (time.time())
    folder = os.path.join(OPT_FOLDER, storeFolder)
    if not os.path.exists(folder):
        os.makedirs(folder)
    out = os.path.join(folder, '%s.jpg' % model)

    dicRet = generate(modelPath, gpu, path, median_filter, padding, out, mode)
    dicRet[RET_MODEL] = model

    log ("processImage done!")

    return dicRet

@app.errorhandler(400)
def not_found(error):
    log('not_found 400')
    return make_response(jsonify( { 'error': 'Bad request' } ), 400)

@app.errorhandler(404)
def not_found(error):
    log('not_found 404')
    return make_response(jsonify( { 'error': 'Not found' } ), 404)


def make_public_task(task):
    new_task = {}
    for field in task:
        if field == 'id':
            new_task['uri'] = url_for('get_task', task_id = task['id'], _external = True)
        else:
            new_task[field] = task[field]
    return new_task


def log(msg):
    print ("[%s] %s" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), msg))
    #print ("[%s] %s" % (, msg))

def genThumb():
    src = "sample_images/tubingen.128.jpg"
    count = 1
    for modelName in PRE_TRAINED_MODELS:
        print ("----- %d ---- %s" % (count, modelName))
        processImage(src, 1, modelName, thumbMode = True)
        count += 1

def __stressTest(*args):
    path = os.path.join("sample_images", "tubingen.jpg")
    mode = 1
    model = "seurat"
    dicRet = processImage(path, mode, model)

def stressTestThreadCount(count):
    import threading
    lstThreads = []
    for i in xrange(1, count+1):
        print("Stresst test .... %d" % i)
        t = threading.Thread(target = __stressTest)
        t.start()
        lstThreads.append(t)

    for t in lstThreads:
        t.join()

def stressTest():
    loop = 10
    for i in xrange(1, loop):
        print("\r\n\r\nStresst test Round %d >>>>>>>>>>>>>>>>>>" % i)
        stressTestThreadCount(i)
        print("Stresst test Round %d Complete !!! " % i)

if __name__ == '__main__':
    import multiprocessing, sys
    log("cpu_count : %d" % multiprocessing.cpu_count())

    isDebug = False
    isStressTest = False
    if len(sys.argv) >= 2:
        isDebug = sys.argv[1] == 'dbg'
        isStressTest = sys.argv[1] == 'stress'
    debug = isDebug
    port = 4000 if isDebug else 5000
    serverMode = "DEBUG" if debug else "PRODUCTION"

    if isStressTest:
        stressTest()
    else:
        if isDebug:
            #processImage('sample_images/tubingen.jpg')
            src = "sample_images/tubingen.jpg"
            model = "model_trained_by_kevin"
            ouput = "sample_images/output.jpg"
            mode = 1

            #processImage(src, mode, model)
            #genThumb()

        log("launch server in %s mode ..." % serverMode)
        app.run(host='0.0.0.0', debug = debug, port = port)
