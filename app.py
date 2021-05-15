from flask import Flask,jsonify,request,render_template
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
     return render_template('index.html')
 
@app.route('/predict',methods=['POST'])
def predict():
		'''
		For rendering results on HTML GUI
		'''
		int_features = [int(x) for x in request.form.values()]
		final_features = [np.array(int_features)]
		#final_features = final_features.reshape(-1,1)
		prediction = model.predict(final_features)
		output = round(prediction[0], 2)

		return render_template('index.html', prediction_text='Estimated salary will be $ {}'.format(output)) 

		
if __name__ == "__main__":
    app.run(debug=True)
    

