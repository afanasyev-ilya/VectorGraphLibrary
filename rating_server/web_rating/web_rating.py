from flask import Flask
from flask import render_template, request, session
from .lib import normalized_rating


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@app.route('/main', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():
    slider_values = {}
    rating_data = normalized_rating.get_list_rating(slider_values)
    return render_template('normalized_rating.html', rating_data=rating_data, slider_vals=None)


@app.route('/update_rating', methods=['GET'])
def update_rating():
    slider_values = request.args

    rating_data = normalized_rating.get_list_rating(slider_values)
    return render_template('normalized_rating.html', rating_data=rating_data, slider_vals=slider_values)
