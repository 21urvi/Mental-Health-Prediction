from flask import Flask, request, render_template
import pickle
import numpy as np
import os

app = Flask(__name__, template_folder='templates')

# Check if the model file exists before loading
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Model file '{model_path}' not found!")

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input features from the form
        form_data = request.form.values()
        int_features = []

        # Convert input values to integers and handle errors
        for val in form_data:
            try:
                int_features.append(int(val))
            except ValueError:
                return render_template('index.html', pred="Error: Please enter valid numeric values.")

        # Get the expected number of features
        expected_features = getattr(model, "n_features_in_", None)

        if expected_features is None and hasattr(model, "estimators_"):  # Stacking Classifier Fix
            expected_features = model.estimators_[0].n_features_in_

        # Ensure the correct number of inputs
        if expected_features and len(int_features) != expected_features:
            return render_template('index.html', pred=f"Error: Model expects {expected_features} inputs, but received {len(int_features)}.")

        # Convert input features to NumPy array and reshape
        final_input = np.array(int_features).reshape(1, -1)
        print(f"Received Features: {int_features}")
        print(f"Reshaped Input: {final_input}")

        # Check if model supports probability prediction
        if hasattr(model, 'predict_proba'):
            prediction = model.predict_proba(final_input)
            output = '{0:.2f}'.format(prediction[0][1])
        else:
            prediction = model.predict(final_input)
            output = str(prediction[0])  # Use direct prediction if `predict_proba` is unavailable

        # Determine treatment need based on threshold
        if float(output) > 0.5:
            result_text = f'You need treatment.\nProbability of mental illness: {output}'
        else:
            result_text = f'You do not need treatment.\nProbability of mental illness: {output}'

        return render_template('index.html', pred=result_text)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template('index.html', pred="An error occurred while making the prediction. Please try again.")

if __name__ == '__main__':
    app.run(debug=True)
