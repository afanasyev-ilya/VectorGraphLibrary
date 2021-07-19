from flask import Flask
from flask import render_template, request, session
from .lib import normalized_rating


app = Flask(__name__)


@app.route('/')
@app.route('/main')
@app.route('/index')
def home():
    rating_data = normalized_rating.get_list_rating()
    return render_template('normalized_rating.html', rating_data=rating_data)
