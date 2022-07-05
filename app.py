from flask import Flask,render_template,request , redirect
import numpy as np
import pandas as pd
import pickle
import math


with open('CarPriceModel.pkl','rb') as fb:
    data = pickle.load(fb)
app = Flask(__name__)
car = pd.read_csv("new_clean_car_data")
@app.route('/')
def index():
    companies =sorted(car['company'].unique()) 
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse= True)
    fuel_type = sorted(car['fuel_type'].unique())
    return render_template('index.html' ,   companies = companies, years = year, fuel_types = fuel_type, car_models = car_models)
@app.route('/predict',methods=['POST'])
def predict():
    company= request.form.get('company')
    model = request.form.get('model_name')
    year = int(request.form.get('year'))
    f_type = request.form.get('fuel_type')
    k_traveled = int(request.form.get('kilometre_travel'))
    #print(comapany,model,year,f_type,k_traveled)
    prediction = data.predict(pd.DataFrame([[model,company,year,k_traveled,f_type]],columns=['name','company','year','kms_driven','fuel_type']))
    #data1= data.predict(pd.DataFrame([['Hyundai Santro Xing','Hyundai','2007','45000','Petrol']],columns=['name','company','year','kms_driven','fuel_type']))
  
   # prediction=data.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],data=np.array([model, company,year,k_traveled,f_type]).reshape(1, 5)))
    #print(prediction)
    #print(prediction)

    return str(math.ceil(prediction[0]))

if __name__ == '__main__':
    app.run(debug=True)