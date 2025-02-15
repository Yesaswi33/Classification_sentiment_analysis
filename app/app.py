from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model & vectorizer
with open("models/sentiment_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        review = request.form["review"]
        review_vectorized = vectorizer.transform([review])
        prediction = model.predict(review_vectorized)[0]
        sentiment = "Positive 😊" if prediction == 1 else "Negative 😞"
        return render_template("index.html", sentiment=sentiment, review=review)
    return render_template("index.html", sentiment=None)

if __name__ == "__main__":
    app.run(port=50013, debug=True)

