from flask import Flask, render_template, request
from service import data_process_service
from flask import jsonify

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


@app.route('/data', methods=["POST"])
def match():
    province_name = request.form['province_name']
    city_name = request.form['city_name']
    if city_name == '武汉' or province_name == '武汉':
        true_data = data_process_service.get_true_data("湖北省", "武汉")
        pre_data = data_process_service.get_predict_data("武汉")
    else:
        true_data = data_process_service.get_true_data(province_name, None)
        pre_data = data_process_service.get_predict_data(province_name)
    return jsonify({"true_data": list(zip(true_data[0], true_data[1])), "pre_data": list(zip(pre_data[0], pre_data[1])), 'x': pre_data[0]})


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
