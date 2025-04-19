from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import requests
import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import re

app = Flask(__name__)

# Replace with your Ngrok URL
NGROK_URL = "https://9f78-34-126-86-58.ngrok-free.app/generate"

# Load your fine-tuned BERT model and tokenizer
model = BertForSequenceClassification.from_pretrained('./bert_model/')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

stop_words = set(stopwords.words('english'))
stop_words.add("rt")  # Adding 'rt' to remove retweet in dataset

# Removing Emojis
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# Replacing user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "", raw_text)
    return text

# Removing URLs
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    text = re.sub(url_regex, '', raw_text)
    return text

# Removing Unnecessary Symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')
    text = text.replace(".", '')
    text = text.replace(",", '')
    text = text.replace("#", '')
    text = text.replace(":", '')
    text = text.replace("?", '')
    return text

# Stemming
def stemming(raw_text):
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in raw_text.split()]
    return ' '.join(words)

# Removing stopwords
def remove_stopwords(raw_text):
    tokenize = word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = ' '.join(text)
    return text

def preprocess(data):
    clean = []
    clean = [text.lower() for text in data]
    clean = [change_user(text) for text in clean]
    clean = [remove_entity(text) for text in clean]
    clean = [remove_url(text) for text in clean]
    clean = [remove_noise_symbols(text) for text in clean]
    clean = [stemming(text) for text in clean]
    clean = [remove_stopwords(text) for text in clean]

    return clean

def classify_text(text):
    # Preprocess text
    text_list = [text]
    preprocessed_text = preprocess(text_list)[0]

    # Tokenize the text
    tokenized_input = tokenizer(preprocessed_text, return_tensors='pt')

    # Get model predictions
    output = model(**tokenized_input)
    logits = output.logits

    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    predicted_label = torch.argmax(probabilities, dim=1).item()
    if predicted_label == 0:
        predicted_label = "Appropriate"
    else:
        predicted_label = "Inappropriate"

    probability_class_0 = probabilities[0][0].item()
    probability_class_1 = probabilities[0][1].item()

    # Formulate response
    response = {
        "text": text,
        "predicted_class": predicted_label,
        "probabilities": {
            "appropriate": probability_class_0,
            "inappropriate": probability_class_1
        }
    }
    return response


# Function to send data to the API
def send_data(input_text):
    # Define the payload
    payload = {"input_text": input_text}

    # Send a POST request to the Ngrok URL
    try:
        response = requests.post(NGROK_URL, json=payload, timeout=30)  # Increased timeout if needed
        response.raise_for_status()  # Raise an error if the request fails
        result = response.json()
        print("API response:", result.get('response', result))
        
        # Ensure that 'response' key exists and is a string
        if 'response' in result:
            response_val = result['response']
            # Serialize if it's a dict or list
            if isinstance(response_val, (dict, list)):
                response_val = json.dumps(response_val, indent=2)
            elif not isinstance(response_val, str):
                response_val = str(response_val)
            return {"value": response_val}
        else:
            # If 'response' key is missing, return an error
            return {"error": "Invalid response format from upstream service."}
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    except ValueError:
        # Handles JSON decoding errors
        return {"error": "Invalid JSON response from upstream service."}

@app.route('/rewrite', methods=['GET', 'POST'])
def get_data():
    if request.method == 'POST':
        if 'text' in request.form:
            text = request.form['text']
            print("Received text for moderation:", text)
            result = send_data(text)
            print("Result to send back:", result)
            if 'value' in result:
                return jsonify({"value": result['value']})
            else:
                return jsonify({"error": result.get('error', 'Unknown error')}), 400
        else:
            return jsonify({"error": "Invalid request"}), 400

    return render_template('moderation.html')


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'text' in request.form:
            text = request.form['text']
            text_response = classify_text(text)
            return jsonify(text_response)
        else:
            return jsonify({"error": "Invalid request"}), 400

    return render_template('index5.html')

@app.route('/content-moderation', methods=['GET', 'POST'])
def content_moderation():
    if request.method == 'POST':
        input_text = request.json.get('input_text') if request.is_json else request.form.get('text', '')
        if not input_text or not input_text.strip():
            error_message = "Please enter some text for moderation."
            return jsonify({"error": error_message}), 400

        text_response = classify_text(input_text)
        return jsonify(text_response)

    return render_template('moderation.html')


if __name__ == '__main__':
    app.run(debug=True)
