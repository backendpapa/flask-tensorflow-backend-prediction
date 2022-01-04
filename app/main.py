
import flask
import numpy as np
import tensorflow as tf
from keras.models import load_model
import os
# tf.compat.v1.disable_eager_execution()


app = flask.Flask(__name__)


def getParameters():
    parameters = []
    parameters.append(flask.request.args.get('male'))
    parameters.append(flask.request.args.get('book1'))
    parameters.append(flask.request.args.get('book2'))
    parameters.append(flask.request.args.get('book3'))
    parameters.append(flask.request.args.get('book4'))
    parameters.append(flask.request.args.get('book5'))
    parameters.append(flask.request.args.get('isMarried'))
    parameters.append(flask.request.args.get('isNoble'))
    parameters.append(flask.request.args.get('numDeadRelations'))
    parameters.append(flask.request.args.get('boolDeadRelations'))
    parameters.append(flask.request.args.get('isPopular'))
    parameters.append(flask.request.args.get('popularity'))
    return parameters


def sendResponse(responseObj):
    response = flask.jsonify(responseObj)
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Methods', 'GET')
    response.headers.add('Access-Control-Allow-Headers', 'accept,content-type,Origin,X-Requested-With,Content-Type,access_token,Accept,Authorization,source')
    response.headers.add('Access-Control-Allow-Credentials', True)
    return response

@app.route('/')
def home():
    return "hello models"


@app.route("/predict", methods=["GET"])
def predict():
    global model
    global graph
    MYDIR = os.path.dirname(__file__)
    m_model=os.path.join(MYDIR + "/" +'gotCharactersDeathPredictions.h5' )
    model = load_model(m_model)
    graph = tf.compat.v1.get_default_graph()
    nameOfTheCharacter = flask.request.args.get('name')
    parameters = getParameters()
    inputFeature = np.asarray(parameters).reshape(1, 12)
    with graph.as_default():
        raw_prediction = model.predict(inputFeature)[0][0]
    if raw_prediction > 0.5:
        prediction = 'Alive'
    else:
        prediction = 'Dead'

  
    return sendResponse({nameOfTheCharacter: prediction})







