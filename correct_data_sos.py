#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  9 16:28:48 2020

@author: raphael
"""

import pandas as pd
import numpy as np


data_sos = pd.read_csv('sursaud-corona-quot-dep-2020-07-09-19h15.csv', delimiter = ';')

index = np.where(data_sos['dep'] == 40)[0]
for j in index:
    data_sos['dep'][j] = '40'

print(np.sum(data_sos['dep'] == 40))

data_sos.to_csv('sursaud-corona-quot-dep-2020-07-09-19h15_corrected.csv', sep = ';')