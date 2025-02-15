Sentiment Analysis Web App (Flask + ML)

📌 Project Overview
This is a Flask-based Sentiment Analysis Web Application that classifies user reviews into Positive, Negative, or Neutral using a pre-trained ML model. It utilizes TF-IDF vectorization and a machine learning classifier trained on IMDB reviews to analyze sentiment.

🚀 Features
✔️ User-friendly Web Interface (Flask + HTML/CSS)
✔️ Pre-trained ML Model for sentiment classification
✔️ TF-IDF Vectorization for text processing
✔️ Processes and predicts sentiment in real-time
✔️ API endpoint available for integration

🛠️ Technologies Used

Flask - Web Framework
Scikit-Learn - Machine Learning
NLTK - Natural Language Processing
Pandas & NumPy - Data Processing
Joblib - Model Serialization
Gunicorn - Production Deployment
📂 Project Structure

classification_sentiment_analysis/
│── static/                  # CSS files for UI styling
│── templates/               # HTML templates (Flask frontend)
│── model/                   # Pre-trained sentiment analysis model
│── app.py                   # Main Flask application
│── train_model.py           # Script to train the ML model
│── sentiment_model.pkl      # Trained ML model
│── tfidf_vectorizer.pkl     # TF-IDF vectorizer
│── processed_data.pkl       # Preprocessed dataset
│── IMDB Dataset.csv         # Training dataset
│── requirements.txt         # Dependencies
│── README.md                # Project Documentation
⚙️ Installation & Setup

1️⃣ Clone the Repository
git clone https://github.com/Yesaswi33/Classification_sentiment_analysis.git
cd Classification_sentiment_analysis
2️⃣ Create & Activate a Virtual Environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run the Flask Web App
python app.py
5️⃣ Open in Browser
Go to:
🔗 http://127.0.0.1:5000/

🛠️ Usage

1️⃣ Open the web app and enter a product review.
2️⃣ Click Analyze to get the sentiment result.
3️⃣ The model will classify the sentiment as Positive, Negative, or Neutral.

🌟 Model Training (Optional)

If you want to train your own model, run:

python train_model.py
This script trains a sentiment classification model using the IMDB Dataset and TF-IDF vectorization.

