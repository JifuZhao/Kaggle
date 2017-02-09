#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import pandas as pd

#%%
### Clean the train dataset

data = pd.read_csv('./../../../Data/KaggleData/SanFranciscoCrimeClassification/rawData/train.csv', index_col= ['Dates'], na_values=None,  parse_dates=['Dates'])
data = data[['Category', 'PdDistrict', 'X', 'Y']]

data['Year'] = data.index.year
data['Month'] = data.index.month
data['DayOfWeek'] = data.index.weekday
data['Hours'] = data.index.hour
data['Minutes'] = data.index.minute

data = data[['Year', 'Month', 'DayOfWeek', 'Hours', 'Minutes', 'X', 'Y', 'PdDistrict', 'Category']]

district = list(set(data.PdDistrict.values))
category = list(set(data.Category.values))
district.sort()
category.sort()

i = range(len(district))
j = range(len(category))

district = dict(zip(district, i))
category = dict(zip(category, j))

temp = data.PdDistrict.values
for i in range(len(temp)):
    temp[i] = district[temp[i]]

temp = data.Category.values
for i in range(len(temp)):
    temp[i] = category[temp[i]]

f = open('./../data/Category.txt', 'w')
listCategory = list(category)
listCategory.sort()
for i in range(len(listCategory)):
    f.writelines(listCategory[i] + ' ' + str(i) + '\n')
f.close()

f = open('./../data/PdDistrict.txt', 'w')
listDistrict = list(district)
listDistrict.sort()
for i in range(len(listDistrict)):
    f.writelines(listDistrict[i] + ' ' + str(i) + '\n')
f.close()

data.to_csv('./../data/train_clean.csv', index=False)

#%%
### Clean the test dataset

data = pd.read_csv('./../../../Data/KaggleData/SanFranciscoCrimeClassification/rawData/test.csv', index_col= ['Dates'], na_values=None,  parse_dates=['Dates'])
data = data[['Id', 'PdDistrict', 'X', 'Y']]

data['Year'] = data.index.year 
data['Month'] = data.index.month
data['DayOfWeek'] = data.index.weekday
data['Hours'] = data.index.hour
data['Minutes'] = data.index.minute

data = data[['Id', 'Year', 'Month', 'DayOfWeek', 'Hours', 'Minutes', 'X', 'Y', 'PdDistrict']]

temp = data.PdDistrict.values
for i in range(len(temp)):
    temp[i] = district[temp[i]]

data.to_csv('./../data/test_clean.csv', index=False)




