from flask import Flask, request, jsonify, render_template
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Initialize the lemmatizer and load the model and vectorizer
lm = WordNetLemmatizer()
model = joblib.load('svc_spam_model.pkl')
cv = joblib.load('count_vectorizer.pkl')  # Load the trained CountVectorizer

app = Flask(__name__)

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y.copy()
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    text = y.copy()
    y.clear()
    for i in text:
        y.append(lm.lemmatize(i, pos='v'))
    return " ".join(y)

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    message = data.get('message')

    transformed_message = transform_text(message)
    print(transformed_message)
    input_data_vectorized = cv.transform([transformed_message]).toarray()
    print(input_data_vectorized)
    prediction = model.predict(input_data_vectorized)
    result = 'spam' if prediction[0] == 1 else 'ham'

    return jsonify({'message': result})

if __name__ == '__main__':
    app.run(debug=True)
