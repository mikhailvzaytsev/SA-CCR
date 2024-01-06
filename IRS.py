import yfinance as yf

# import the used libraries
import numpy as np
import pandas as pd
import math
import json
import requests

from urllib import parse

# параметры IRS
key_rate = 0.075
notional = 1000

# получаем данные по RUONIA
url_ruonia = "https://www.cbr.ru/Queries/UniDbQuery/DownloadExcel/125022?FromDate=05%2F13%2F2023&ToDate=05%2F19%2F2023&posted=False"
response = requests.get(url_ruonia)
file_path_ruonia = "/Users/mihailzaytsev/Desktop/practical-python/Risk/data.xlsx"
with open(file_path_ruonia, "wb") as file:
    file.write(response.content)
df = pd.read_excel(file_path_ruonia)
ruonia = df.iloc[0, 3]/100

# получаем дисконтирующую кривую на основе КБД Мосбиржи

gcurve_url = "https://iss.moex.com/iss/engines/stock/zcyc.json"
response = requests.get(gcurve_url)
result = json.loads(response.text)
col_name = result['yearyields']['columns']
resp_date = result['yearyields']['data']
gcurve_data = pd.DataFrame(resp_date, columns = col_name)
discount_curve = np.exp(-gcurve_data['value']/100*gcurve_data['period'])

# проводим интерполяцию дисконт-фактора в полугодовые выплаты по IRS
payment_date = []
i = 0.5
while i <= 5:
    payment_date.append(i)
    i += 0.5
interpolated_discount_curve = np.interp(payment_date, gcurve_data['period'], np.log(discount_curve))
interpolated_discount_curve = np.exp(interpolated_discount_curve)

table2 = pd.DataFrame({'date': payment_date, 'discount_curve': interpolated_discount_curve})

last_row_index = len(table2) - 1
table2['discounted_fixed_CF'] = ((key_rate * notional) / 2) * table2['discount_curve']
table2.loc[last_row_index, 'discounted_fixed_CF'] = ((key_rate * notional) / 2 + notional) * table2.loc[last_row_index, 'discount_curve']

discounted_fixed_leg = 0
for i in table2['discounted_fixed_CF']:
    discounted_fixed_leg += i

discounted_float_leg = (ruonia/2 * notional + notional) * table2['discount_curve'][0]

print(table2)
print("key rate: ", key_rate)
print("RUONIA: ", ruonia)
print("Fixed leg value: ", discounted_fixed_leg)
print("Float leg value: ", discounted_float_leg)
print("Value of the IRS: ", (discounted_float_leg - discounted_fixed_leg))