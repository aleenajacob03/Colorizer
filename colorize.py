import numpy as np,cv2,os,io,base64

DIR = r"C:\Users\USER\VSCode\Extra\GenAI"
PROTOTXT = os.path.join(DIR, r"models/colorization_deploy_v2.prototxt")
POINTS = os.path.join(DIR, r"models/pts_in_hull.npy")
MODEL = os.path.join(DIR, r"models/colorization_release_v2.caffemodel")

net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
pts = np.load(POINTS)
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2, 313, 1, 1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

from flask import Flask, render_template, request
UPLOAD_FOLDER = '/Users/USER/VSCode/Extra/GenAI/static/images'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'Hello'
 
 
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('image.html')
                       
@app.route('/',  methods=("POST", "GET"))
def uploadFile():
    if request.method == 'POST':
        image = cv2.imdecode(np.frombuffer(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        scaled = image.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (image.shape[1], image.shape[0]))
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
        colorized = (255 * colorized).astype("uint8")
        data = io.BytesIO()
        imenc = cv2.imencode(".jpg", colorized)[1]
        imb = base64.b64encode(imenc)
        return render_template('show.html', img=imb.decode('utf-8'))

if __name__=='__main__':
    app.run(debug = True)