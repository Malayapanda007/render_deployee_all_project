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




import numpy as np

# Load pickled data
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

# app = Flask(__name__)

# # Define the home route
@app.route('/movie_recommendation')
def movie_recommendation_index():
    return render_template('index1.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num-Rating'].values),
                           rating=list(popular_df['avg-Rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_books', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')

    # Check if the book exists in the pivot table
    if user_input in pt.index:
        index = np.where(pt.index == user_input)[0][0]  # Get the index of the user input

        # Get similar items based on the similarity scores
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)

        return render_template('recommend.html', data=data)

    else:
        # Return an error message if the book name is not found
        return render_template('index1.html',
                           book_name=list(popular_df['Book-Title'].values),
                           author=list(popular_df['Book-Author'].values),
                           image=list(popular_df['Image-URL-M'].values),
                           votes=list(popular_df['num-Rating'].values),
                           rating=list(popular_df['avg-Rating'].values)
                           )

if __name__ == "__main__":

    app.run(debug=True)


