from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    inputs = {
        'Gender': request.form['Gender'],
        'Married': request.form['Married'],
        'Dependents': request.form['Dependents'],
        'Education': request.form['Education'],
        'Self_Employed': request.form['Self_Employed'],
        'ApplicantIncome': request.form['ApplicantIncome'],
        'CoapplicantIncome': request.form['CoapplicantIncome'],
        'LoanAmount': request.form['LoanAmount'],
        'Loan_Amount_Term': request.form['Loan_Amount_Term'],
        'Credit_History': request.form['Credit_History'],
        'Property_Area': request.form['Property_Area']
    }
    features = np.array([[float(v) for v in inputs.values()]])
    prediction = model.predict(features)[0]

    result = 'Loan Approved' if prediction == 1 else 'Loan Rejected'
    return render_template('index.html', result=result, **inputs)

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT",10000))
    app.run(host='0.0.0.0',port=port)
