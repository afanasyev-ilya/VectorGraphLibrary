from flask import Flask
from flask import render_template, request, session
from .lib import normalized_rating
import pprint

def get_value(rating):
    return (float)(rating['rating'])

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
@app.route('/main', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def home():

    slider_values = request.args

#    print("\n-----------------------------------------------------------------\n")
#    print(slider_values)
#    print("\n-----------------------------------------------------------------\n")

    rating_data = normalized_rating.get_list_rating(slider_values)
    print("\n-----------------------------------------------------------------\n")
    print(rating_data)
    print("\n-----------------------------------------------------------------\n")

    rating_data.sort(key=get_value, reverse=True)


    full_perf_table = normalized_rating.get_perf_table()
    for i in range(0, len(rating_data)):
        rating_data[i]["pos"] = str(i + 1)

    return render_template('normalized_rating.html', rating_data=rating_data, slider_vals=slider_values,
                           full_perf_table=full_perf_table)
