# app.py #
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

#  load the pre-trained model and other necessary components  #
classifier_top_features = joblib.load(r'C:\Users\amina\Desktop\Python_course\Projet_flask\classifier_top_features.joblib')
#top_feature_indices = np.load(r'C:\Users\amina\Desktop\Python_course\Projet_flask\top_feature_indices.npy')

@app.route('/', methods=['GET', 'POST']) #get et post se trouve dans request
def index():
    if request.method == 'POST':
        #retrieve user inputs
        user_inputs = [float(request.form[f'input{i+1}']) for i in range(10)] #List comprehension 

        #prediction using the model
        prediction = classifier_top_features.predict(np.array([user_inputs])) #transforme le user imput en array pour prédire le résultat

        return render_template('result.html', prediction=prediction[0]) #page de prédiction result html

    return render_template('index.html')#page d'acceuil index.html (si la condition n'est pas remplie on reste sur la page d'acceuil)

if __name__ == '__main__':
    app.run(debug=True)
