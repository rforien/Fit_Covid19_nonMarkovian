#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 22:14:17 2020

@author: raphael
"""

import pandas as pd
import numpy as np

data = pd.read_csv('donnees-hospitalieres-covid19-2020-07-01-19h00.csv', delimiter = ';')

dates = ['27/06/2020', '28/06/2020', '29/06/2020']
correct_dates = ['2020-06-27', '2020-06-28', '2020-06-29']

for (i, date) in enumerate(dates):
    index = np.where(data['jour'] == date)
    for j in index:
        data['jour'][j] = correct_dates[i]
        
print(np.unique(data['jour']))

data.to_csv('donnees-hospitalieres-covid19-2020-07-01-19h00_corrected.csv', sep = ';')