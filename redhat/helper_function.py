#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__author__      = "Jifu Zhao"
__email__       = "jzhao59@illinois.edu"
__date__        = "mm/dd/2016"
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# function to process the dataset
def process(fname, people_path='./data/people.csv'):
    # load the dataset
    activity = pd.read_csv(fname, parse_dates=['date'], dtype={'activity_id': np.str})
    people = pd.read_csv(people_path, parse_dates=['date'])

    # fill NaN values in activity
    activity = activity.fillna('type 0')
    # rename column names in people
    people.columns = people.columns.map(lambda col: col if 'char' not in col
                                                        else col[:5] + str(int(col[5:]) + 10))
    
    # process the date in people
    activity['activity_year'] = activity['date'].dt.year
    activity['activity_month'] = activity['date'].dt.month
    activity['activity_day'] = activity['date'].dt.day
    activity['activity_isWeekend'] = (activity['date'].dt.weekday >= 5)
    
    # process the date in people
    people['people_year'] = people['date'].dt.year
    people['people_month'] = people['date'].dt.month
    people['people_day'] = people['date'].dt.day
    people['people_isWeekend'] = (people['date'].dt.weekday >= 5)
    
    # drop the date column in people and activity
    activity = activity.drop('date', axis=1)
    people = people.drop('date', axis=1)
    
    # merge the activity and people files
    data = pd.merge(activity, people, how='left', on=['people_id'])
    # delete to save memory
    del [people, activity]

    index = data['activity_id']
    # drop columns
    data.drop(['people_id', 'activity_id'], axis=1, inplace=True)

    # delete useless words
    columns = ['char_' + str(i) for i in range(1, 20)] + ['activity_category', 'group_1']
    for i in columns:
        data[i] = data[i].apply(lambda x: x.split(' ')[1])

    data = data.astype(int)

    return index, data


# see the number of unique values for each column
def uniqueValue(data, threshold=100):
    """ function to find the unique values """
    for i in data.columns:
        unique = data[i].unique()
        if i == 'outcome':
            continue
        if len(unique) < threshold:
            print(i, len(unique), sorted(unique))
        else:
            print(i, len(unique))
            visualizeDist(data, i, bins=len(unique), figsize=(8, 3))


def visualizeDist(data, feature, bins=2, figsize=(8, 3)):
    """ function to visualize the distribution of the feature """
    positive = data[data['outcome'] == 1]
    negative = data[data['outcome'] == 0]
    
    pos_value, pos_index = np.histogram(positive[feature].values, bins=min(bins, 100))
    neg_value, neg_index = np.histogram(negative[feature].values, bins=min(bins, 100))
    width = (pos_index[1] - pos_index[0]) / 1.5
    
    plt.figure(figsize=figsize)
    plt.title("Distribution of " + feature)
    plt.bar(pos_index[:-1], pos_value, width=width, color='r', alpha=0.4, label='Positive')
    plt.bar(neg_index[:-1], neg_value, width=width, color='g', alpha=0.4, label='Negative')
    plt.yscale('log')
    plt.legend()
    plt.show()
