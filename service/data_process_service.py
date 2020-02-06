import datetime
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

start_date = datetime.datetime.strptime('2020-01-24', '%Y-%m-%d')
all_days = 90
date_list = [(start_date + datetime.timedelta(days=i)
              ).strftime("%Y-%m-%d") for i in range(all_days)]
x = np.arange(all_days)+1

data_path = os.path.join(os.path.dirname(
    __file__), '../data/DXYArea.csv')
data = pd.read_csv(data_path)
data['time_index'] = data.apply(lambda x: (datetime.datetime.strptime(
    x['updateTime'][:10], '%Y-%m-%d')-start_date).days+1, axis=1)
# print(data[data['provinceName'] == '湖北省'])

paras_data_path = os.path.join(
    os.path.dirname(__file__), '../data/result.json')
with open(paras_data_path) as f:
    paras_data = json.load(f)

data['pro_num'] = data['province_confirmedCount'] - \
    data['province_curedCount'] - data['province_deadCount']
data['city_num'] = data['city_confirmedCount'] - \
    data['city_curedCount'] - data['city_deadCount']


def get_true_data(province_name, city_name):
    if province_name is None:
        return None
    if city_name is None:
        province_data = data[data['provinceName'] == province_name]
        province_data_new = province_data.groupby(
            'time_index')['pro_num'].max().reset_index()
        time_list = list(province_data_new['time_index'])
        num_list = list(province_data_new['pro_num'])
    else:
        city_data = data[(data['cityName'] == city_name) &
                         (data['provinceName'] == province_name)]
        city_data_new = city_data.groupby(
            'time_index')['city_num'].max().reset_index()
        time_list = list(city_data_new['time_index'])
        num_list = list(city_data_new['city_num'])
    return [date_list[i-1] for i in time_list], num_list


def get_predict_data(province_name):
    paras = paras_data.get(province_name)
    if paras is None:
        return
    predict_data = func(params=paras, x1=x)
    predict_data = [int(x) for x in predict_data if x > 0]
    return date_list[:len(predict_data)], predict_data


def func(params, x1):
    a1, b1, c1, d1 = params
    return a1 * np.exp(-(x1 - b1) ** 2 / (2 * c1 ** 2)) + d1
