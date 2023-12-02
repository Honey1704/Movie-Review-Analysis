import pickle
from flask import Flask, request, jsonify, render_template

# Initialize the Flask app
app = Flask(__name__)

# Load the pre-trained model and Bag of Words vocabulary
model = pickle.load(open('model1.pkl', 'rb'))
word_dict = pickle.load(open('bow.pkl', 'rb'))

# Define a function to preprocess and predict sentiment
def predict_sentiment(text):
    # Preprocess the input text
    cleaned_text = clean(text)
    special_chars_removed = is_special(cleaned_text)
    text_lower = to_lower(special_chars_removed)
    text_no_stopwords = rem_stopwords(text_lower)
    text_stemmed = stem_txt(text_no_stopwords)
    
    # Calculate the Bag of Words representation for the input text
    input_features = []
    for word in word_dict:
        input_features.append(text_stemmed.count(word))
    
    # Make a sentiment prediction using the pre-trained model
    sentiment = model.predict([input_features])[0]
    return sentiment

# Define a route for the sentiment analysis web page
@app.route('/')
def sentiment_analysis():
    return render_template('index.html')

# Define a route for handling sentiment prediction
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['user_input']
        sentiment = predict_sentiment(user_input)
        result = 'Positive' if sentiment == 1 else 'Negative'
        return jsonify({'sentiment': result})

if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080)
