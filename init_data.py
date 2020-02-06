import datetime
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import leastsq
from tqdm import tqdm

start_date = datetime.datetime.strptime('2020-01-24', '%Y-%m-%d')

data_path = os.path.join(os.path.dirname(
    __file__), './data/DXYArea.csv')
data = pd.read_csv(data_path)
data['time_index'] = data.apply(lambda x: (datetime.datetime.strptime(
    x['updateTime'][:10], '%Y-%m-%d')-start_date).days+1, axis=1)
# print(data[data['provinceName'] == '湖北省'])

data['pro_num'] = data['province_confirmedCount'] - \
    data['province_curedCount'] - data['province_deadCount']
data['city_num'] = data['city_confirmedCount'] - \
    data['city_curedCount'] - data['city_deadCount']


def get_province_data(province_name, city_name):
    if province_name is None:
        return None
    if city_name is None:
        province_data = data[data['provinceName'] == province_name]
        province_data_new = province_data.groupby(
            'time_index')['pro_num'].max().reset_index()
        return province_data_new[['time_index', 'pro_num']]
    else:
        city_data = data[(data['cityName'] == city_name) &
                         (data['provinceName'] == province_name)]
        city_data_new = city_data.groupby(
            'time_index')['city_num'].max().reset_index()
        return city_data_new[['time_index', 'city_num']]


def func(params, x1):
    a1, b1, c1, d1 = params
    return a1 * np.exp(-(x1 - b1) ** 2 / (2 * c1 ** 2)) + d1


def error(params, x1, y1):
    return func(params, x1) - y1


def slove_para(x, y):
    p0 = (10, 10, 10, 10)
    paras = leastsq(error, p0, args=(x, y))
    return paras


province_names = list(data['provinceName'].drop_duplicates())
city_names = data[['provinceName', 'cityName']].drop_duplicates().values
result_data = {}
for province_name in tqdm(province_names):
    one_province_data = get_province_data(province_name, None)
    if one_province_data.shape[0] < 9 or one_province_data['pro_num'].max() < 130:
        continue
    a, b, c, d = slove_para(
        one_province_data['time_index'], one_province_data['pro_num'])[0]
    result_data[province_name] = [a, b, c, d]

# TODO 加入各县市数据

# 加入武汉市
one_province_data = get_province_data("湖北省", "武汉")
a, b, c, d = slove_para(one_province_data['time_index'], one_province_data['city_num'])[0]
result_data['武汉'] = [a, b, c, d]

with open('./data/result.json', 'w') as f:
    json.dump(result_data, f, ensure_ascii=False)
# print(get_province_data('湖北省', None))
