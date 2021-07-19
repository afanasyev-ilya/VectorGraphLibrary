from flask import Flask
from .lib import normalized_rating

app = Flask(__name__)


@app.route("/")
def hello_world():
    text = normalized_rating.get_text_rating()
    return "<p>" + text + "</p>"
