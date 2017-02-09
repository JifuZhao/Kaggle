#!/usr/bin/env python

# -*- coding: utf-8 -*-
"""
@author: Jifu Zhao
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

#%%
def dataGroup(data, columnName):
    result = pd.DataFrame()
    item = set(data[columnName])
    for i in item:
        temp = data[data[columnName] == i].groupby('Category', sort=True).count()
        df = pd.DataFrame(temp[columnName])
        df = df.rename(columns = {columnName: i})
        result = result.join(df, how='outer')
        
    result = result.fillna(0)
    result.index.name = ''    
    result['Category'] = result.index
    result.reindex(columns=list(item) + ['Category'])

    return result

def categoryTrans():
    nameList = []
    with open('./../data/CategoryName.txt', 'r') as f:
        for line in f:
            nameList.append(line[:-1])
            
    return nameList
    
def districtTrans():
    nameList = []
    with open('./../data/PdDistrictName.txt', 'r') as f:
        for line in f:
            nameList.append(line[:-1])
            
    return nameList
    
#%%
trainPath = './../data/train_clean.csv'
trainData = pd.read_csv(trainPath)

#%%
year = dataGroup(trainData, 'Year')
month = dataGroup(trainData, 'Month')
week = dataGroup(trainData, 'DayOfWeek')
hour = dataGroup(trainData, 'Hours')
PdDistrict = dataGroup(trainData, 'PdDistrict')

year['Category'] = categoryTrans()
month['Category'] = categoryTrans()
week['Category'] = categoryTrans()
hour['Category'] = categoryTrans()
PdDistrict['Category'] = categoryTrans()

#%% plot barplot for different years
N = 2015 - 2003 + 1
fig, ax = plt.subplots(1, N, figsize=(20, 10), sharey = True)
for i in range(N):
    sns.barplot(x=(1+2003), y='Category', data=year, ax=ax[i])
    ax[i].set(xticks = [7000, 14000], xlabel = "", ylabel = "", title = str(2003+i))
    
plt.suptitle('Change over Years', size=16)    
plt.savefig('./figure/year.eps')

#%% plot barplot for different months
N = 12
monthList = calendar.month_name[1:13]
fig, ax = plt.subplots(1, N, figsize=(20, 10), sharey = True)
for i in range(N):
    sns.barplot(x=i+1, y='Category', data=month, ax=ax[i])
    ax[i].set(xticks = [9000, 18000], xlabel = "", ylabel = "", title = monthList[i])

plt.suptitle('Change over Months', size=16)    
plt.savefig('./figure/month.eps')

#%% plot barplot for different weeks
N = 7
weekList = calendar.day_name[:]
fig, ax = plt.subplots(1, N, figsize=(20, 10), sharey = True)
for i in range(N):
    sns.barplot(x=i, y='Category', data=week, ax=ax[i])
    ax[i].set(xticks = [10000, 20000, 30000], xlabel = "", ylabel = "", title = weekList[i])

plt.suptitle('Change over Weeks', size=16)    
plt.savefig('./figure/week.eps')

#%% plot barplot for different hours
N = 24
fig, ax = plt.subplots(1, N, figsize=(30, 10), sharey = True)
for i in range(N):
    sns.barplot(x=i, y='Category', data=hour, ax=ax[i])
    ax[i].set(xticks = [7000, 14000], xlabel = "", ylabel = "", title = str(i))

plt.suptitle('Change over Hours', size=16)    
plt.savefig('./figure/hour.eps')

#%% plot barplot for different PdDistricts
N = 10
PdDistrictList = districtTrans()
fig, ax = plt.subplots(1, N, figsize=(20, 10), sharey = True)
for i in range(N):
    sns.barplot(x=i, y='Category', data=PdDistrict, ax=ax[i])
    ax[i].set(xticks = [22000, 45000], xlabel = "", ylabel = "", title = PdDistrictList[i])

plt.suptitle('Change over PdDistricts', size=16)
plt.savefig('./figure/PdDistrict.eps')

