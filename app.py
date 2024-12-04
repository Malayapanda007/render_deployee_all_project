from flask import Flask,render_template,request,redirect
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


from db import Database
# import prediction

dbo=Database()


app=Flask(__name__)

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/registration')
def registration(): 
    return render_template('registration.html')


@app.route('/perform_registration', methods=['POST'])
def perform_registration(): 
    name = request.form.get('name')
    email = request.form.get('email')
    password = request.form.get('password')

    response=dbo.insert(name,email,password)

    if response:
        return render_template('login.html',message='Registration Successful')
    else:
        return render_template('registration.html',message='Email Already Exist')



@app.route('/perform_login', methods=['POST'])
def perform_login():
    email = request.form.get('email')
    password = request.form.get('password')

    response=dbo.search(email,password)

    if response:
        return redirect('/profile')
    else:
        return render_template('login.html',message='Incorrect Email/password')

@app.route('/profile')
def profile():
    return render_template('home.html')



@app.route('/email_spam')
def email_spam(): 
    return redirect('/spam_sms')

@app.route('/spam_sms')
def spam_sms():
    return render_template('email_spam.html')






# Download necessary NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load the trained model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Initialize the Flask app


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
    
    return render_template('email_spam.html', prediction_text='Prediction: {}'.format(result))

if __name__ == "__main__":

    app.run(debug=True)


