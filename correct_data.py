#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:14:17 2020

@author: raphael
"""

import pandas as pd
import numpy as np

data = pd.read_csv('donnees-hospitalieres-covid19-2020-07-09-19h00.csv', delimiter = ';')

# correct date format

dates = ['27/06/2020', '28/06/2020', '29/06/2020']
correct_dates = ['2020-06-27', '2020-06-28', '2020-06-29']

for (i, date) in enumerate(dates):
    index = np.where(data['jour'] == date)[0]
    for j in index:
#        print(j)
        # print(data['dep'].values[j], data['jour'].values[j])
        data['jour'][j] = correct_dates[i]
        
print(np.unique(data['jour']))

# remove duplicate entries for 2020-07-03
index = np.where(data['jour'] == '2020-07-03')[0]
if np.size(index) == 6*101:
    print('remove duplicate')
    k = int(np.size(index)/2)
    data = data.drop(index[k:])

data.to_csv('donnees-hospitalieres-covid19-2020-07-09-19h00_corrected.csv', sep = ';')
