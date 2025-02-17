from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_bcrypt import Bcrypt
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
from sklearn.feature_extraction.text import CountVectorizer
import os
import re
import pickle

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")
bcrypt = Bcrypt(app)

client = MongoClient(os.getenv("MONGO_URI"))
db = client['stress_db']
users_collection = db['users']
messages_collection = db['messages']
text_collection = db['stress_analysis']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_text', methods=['GET', 'POST'])
def analyze_text():
    import datetime
    if 'username' not in session:
        flash('Please log in to access this feature.')
        return redirect(url_for('login'))
    if request.method == 'POST':
        user_text = request.form['text']

        import pandas as pd
        import numpy as np
        data = pd.read_csv("./Dataset/stress.csv")
        import nltk
        import re
        nltk.download('stopwords')
        stemmer = nltk.SnowballStemmer("english")
        from nltk.corpus import stopwords
        import string
        stopword=set(stopwords.words('english'))

        def clean(text):
            text = str(text).lower()
            text = re.sub('\[.*?\]', '', text)
            text = re.sub('https?://\S+|www\.\S+', '', text)
            text = re.sub('<.*?>+', '', text)
            text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
            text = re.sub('\n', '', text)
            text = re.sub('\w*\d\w*', '', text)
            text = [word for word in text.split(' ') if word not in stopword]
            text=" ".join(text)
            text = [stemmer.stem(word) for word in text.split(' ')]
            text=" ".join(text)
            return text
        data["text"] = data["text"].apply(clean)
        x = np.array(data["text"])
        cv = CountVectorizer()
        cv.fit_transform(x)

        label = "No Stress"
        with open('stressModel.pkl', 'rb') as f:
            model = pickle.load(f)
            user = user_text
            data = cv.transform([user]).toarray()
            output = model.predict(data)
            label = output[0]


        analysis_data = {
                "username": session['username'],
                "text": user_text,
                "stress_label": label,
                "analysis_date": datetime.datetime.now().isoformat()
            }
        text_collection.insert_one(analysis_data)


        return render_template('result.html', label=label, user_text=user_text)
    
    return render_template('analyze_text.html')

@app.route('/historical_analysis')
def historical_analysis():
    if 'username' not in session:
        return redirect(url_for('login'))
   
    analyses = text_collection.find({"username": session['username']}).sort("analysis_date")
    return render_template('historical_analysis.html', analyses=analyses)


@app.route('/community', methods=['GET', 'POST'])
def community():
    from datetime import datetime
    if 'username' not in session:
        flash('Please log in to access the community chat.')
        return redirect(url_for('login'))

    if request.method == 'POST':
        message_content = request.form['message']
        username = session['username']
        timestamp = datetime.utcnow()

        if message_content.strip():
            message = {
                'username': username,
                'message': message_content,
                'timestamp': timestamp
            }
            messages_collection.insert_one(message)

    messages = list(messages_collection.find().sort('timestamp', 1))
    return render_template('community.html', messages=messages)

@app.route("/mental_resources")
def resources():
    return render_template('mental_resources.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if users_collection.find_one({'username': username}):
            flash('Error: Username already exists. Please choose a different one.', 'error')
        else:
            hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
            users_collection.insert_one({'username': username, 'password': hashed_password})
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = users_collection.find_one({'username': username})
        if user is None:
            flash('Error: Username does not exist.', 'error')
        elif not bcrypt.check_password_hash(user['password'], password):
            flash('Error: Incorrect password.', 'error')
        else:
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
    
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    flash('You have been logged out.')
    return redirect(url_for('index'))


if __name__ == '__main__':
    app.run(debug=True)