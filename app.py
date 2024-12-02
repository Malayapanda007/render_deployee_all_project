from flask import Flask, request, render_template
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the trained model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Initialize the Flask app
app = Flask(__name__)

# Initialize the PorterStemmer
ps = PorterStemmer()

# Define the preprocessing function
def transform_text(text):
    text = text.lower()  # Convert to lowercase
    text = nltk.word_tokenize(text)  # Tokenize text

    y = []
    for i in text:
        if i.isalnum():  # Keep only alphanumeric characters
            y.append(i)

    text = y[:]
    y.clear()

    # Remove stopwords and punctuation
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    # Apply stemming
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)  # Return the preprocessed text

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Check if the form contains 'message' key (for text input)
    if 'message' in request.form:
        # Preprocess the input message
        input_message = request.form['message']
        processed_message = transform_text(input_message)
        
        # Transform the processed message using the loaded vectorizer
        transformed_message = tfidf.transform([processed_message])
        
        # Make prediction
        prediction = model.predict(transformed_message)
        
        result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    
    # Render the prediction result
    
    return render_template('index.html', prediction_text='Prediction: {}'.format(result))

if __name__ == "__main__":
    app.run(debug=True)
