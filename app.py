from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Capture form data
    features = [float(x) for x in request.form.values()]
    input_data = np.array([features])

    # Get prediction
    prediction = model.predict(input_data)

    # Display result
    result = 'Diabetes' if prediction[0] == 1 else 'No Diabetes'
    return render_template('index.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)
