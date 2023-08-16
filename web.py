from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
model=pickle.load(open('model.pkl','rb'))
onehot_encoder = pickle.load(open('encoder.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/prediction',methods=['POST'])
def predict():

    age = float(request.form.get('Age'))
    base_pay = float(request.form['Base_pay'])
    months = float(request.form['Months'])
    total_sales = float(request.form['Total_Sales'])
    volume = float(request.form['Volume'])
    openingbalance = float(request.form['openingbalance'])
    closingbalance = float(request.form['closingbalance'])

    int_features = [age, base_pay, months, total_sales, volume, openingbalance, closingbalance]
    features = np.array(int_features).reshape(1, -1)
 
    high_school_, intermediate_, pg_ = 0, 0, 0
    if request.form['Education'] == 'High-School': high_school_= 1
    if request.form['Education'] == 'Intermediate': intermediate_= 1
    if request.form['Education'] == 'PG': pg_= 1

    gender_ = 0
    if request.form['Gender'] == 'Male': gender_= 1

    type_one, type_two = 0, 0
    if request.form['Type'] == 'One-year': type_one= 1
    if request.form['Type'] == 'Two-year': type_two= 1

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)

    features = np.append(scaled_features, [high_school_, intermediate_, pg_, gender_, type_one, type_two]).reshape(1, -1)

    
    prediction = model.predict(features)

    output = round(prediction[0], 2)
    return render_template ('result.html',prediction_text="Congrats!!...You are eligible for a Salary of Rs.{}".format(output))

if __name__=='__main__':
    app.run()




