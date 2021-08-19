from flask import Flask
from flask import render_template, request, session
from .lib import normalized_rating
import pprint


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
@app.route('/main', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():

    slider_values = request.args

    print("\n-----------------------------------------------------------------\n")
    print(slider_values)
    print("\n-----------------------------------------------------------------\n")

    rating_data = normalized_rating.get_list_rating(slider_values)
    print("\n-----------------------------------------------------------------\n")
    print(rating_data)
    print("\n-----------------------------------------------------------------\n")
    
    full_perf_table = normalized_rating.get_perf_table()
    pprint.pprint(full_perf_table)
    return render_template('normalized_rating.html', rating_data=rating_data, slider_vals=slider_values,
                           full_perf_table=full_perf_table)
