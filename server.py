#!flask/bin/python
import os, time
import threading
import random
import ServerResponseDef
from time import gmtime, localtime, strftime
from flask import Flask, jsonify, abort, request, make_response, url_for
from flask import send_from_directory
from generate2 import generate
#from threading import Thread

#logging.Formatter(fmt='%(asctime)s.%(msecs)03d',datefmt='%Y-%m-%d,%H:%M:%S')
#logger = logging.getLogger(__name__)

app = Flask(__name__, static_url_path = "")

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'uploads/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


FILE_UPLOAD_PATH = "uploads"
FILE_TENSOR_GRAPH_PATH = "classify_image_graph_def.pb"
OPT_FOLDER = "opt"


PRE_TRAINED_MODELS = [
    "composition", "seurat", "candy_512_2_49000"
    "fur_0", "kanagawa", "scream-style", 
    "cubist", "hokusai", "kandinsky_e2_crop512", "starry"
    "edtaonisl", "hundertwasser", "kandinsky_e2_full512", "starrynight"]

DEFAULT_MODEL = "starry"
MODEL_PATH1 = "models"
MODEL_PATH2 = "/work/machine_learning/prisma_style/open-source-proj/gafr/chainer-fast-neuralstyle-models/models"

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

@app.route('/opt/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    uploads = os.path.join(app.root_path, OPT_FOLDER)
    return send_from_directory(directory=uploads, filename=filename)

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
        print ("\r\n ====================== upload task =====================")
        # Get file object from field of file
        f = request.files['userfile']
        path = os.path.join(FILE_UPLOAD_PATH, f.filename)
        f.save(path)
        # Get str object from field of text
        #s = request.form['name'] + "\r\n"
        print ("request:", request.form)
        mode = int(request.form['mode'])
        print ("mode : ", mode)
        randomModel = PRE_TRAINED_MODELS[random.randint(0, len(PRE_TRAINED_MODELS))]
        model = request.form['model'] if request.form.has_key("model") else randomModel#DEFAULT_MODEL
        log = os.path.join(FILE_UPLOAD_PATH, "log.txt")
        with open(log, 'a') as f:
            f.write("[%s]%s" % (strftime("%Y-%m-%d %H:%M:%S", localtime()), path))
        f.close()

        dicRet = processImage(path, mode, model)
        dicRet["code"] = ServerResponseDef.SUCCESS

        print (dicRet)
        return str(dicRet)

    log('ERR_CMD_NOT_SUPPORT')
    return '{ret:fail, code:%d}' % (ServerResponseDef.ERR_CMD_NOT_SUPPORT)

def processImage(path, mode, model):
    log ("processImage:%s" % path)

    modelName = "%s.model" % model
    modelPath = os.path.join(MODEL_PATH1, modelName)
    if not os.path.exists(modelPath):
        modelPath = os.path.join(MODEL_PATH2, modelName)
    if os.path.exists(modelPath):
        log("Use %s" % modelName)
    else:
        log("[ERROR] model %s is not exist" % modelPath)
        return processImage(path, mode, PRE_TRAINED_MODELS[random.randint(0, len(PRE_TRAINED_MODELS))])

    gpu = -1
    median_filter = 3
    padding = 50
    folder = os.path.join(OPT_FOLDER, "%d" % time.time())
    if not os.path.exists(folder):
        os.makedirs(folder)
    out = os.path.join(folder, 'out.jpg')

    ret = generate(modelPath, gpu, path, median_filter, padding, out, mode)

    log ("processImage done!")

    return ret

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
    #logger.info(msg)

if __name__ == '__main__':
    import multiprocessing, sys
    log("cpu_count : %d" % multiprocessing.cpu_count())

    isDebug = False
    if len(sys.argv) >= 2:
        isDebug = sys.argv[1] == 'dbg'
    debug = isDebug
    port = 4000 if isDebug else 5000
    serverMode = "DEBUG" if debug else "PRODUCTION"

    if isDebug:
        #processImage('sample_images/tubingen.jpg')
        src = "sample_images/tubingen.jpg"
        model = "models/composition.model"
        ouput = "sample_images/output.jpg"
        mode = "1"

        #testGenerate()

    log("launch server in %s mode ..." % serverMode)
    app.run(host='0.0.0.0', debug = debug, port = port)

