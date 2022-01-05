import pickle
#import joblib
import numpy as np
from flask import Flask, request, render_template
app = Flask(__name__)

@app.route('/')
def home():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' For rendering result in GUI'''
    model = pickle.load(open('heart_disease_predictor.pkl','rb'))
    #model = joblib.load('heart_disease_predictor.ml')
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    status = model.predict(final_features)
    return str(outcome(status[0]))

def outcome(status):
    if status == '1':
        return render_template('index.html', response_status = f'Response Status: {status}', prediction_text="Test Result: Positive")
    else:
        return render_template('index.html', response_status = f'Response Status: {status}', prediction_text="Test Result: Negative")

if __name__ == '__main__':
  app.run(debug=True)