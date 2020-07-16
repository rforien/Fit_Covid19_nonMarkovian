#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:22:30 2020

@author: raphael
"""

import pandas as pd
import numpy as np

dep_file = 'departements.csv'

def patches_from_regions(regions):
    deps = pd.read_csv(dep_file, delimiter = ';', index_col = 'departement')
    n = np.size(regions, axis = 0)
    patches = n*[[]]
    sizes = np.zeros(n)
    for (i, region) in enumerate(regions):
        assert region in deps['region'].values
        patches[i] = deps.index[deps['region'] == region].values
        sizes[i] = deps['population'][patches[i]].sum()
    return patches, sizes

def patches_from_region_list(region_list):
    deps = pd.read_csv(dep_file, delimiter = ';', index_col = 'departement')
    n = np.size(region_list, axis = 0)
    patches = n*[[]]
    sizes = np.zeros(n)
    for (i, reg_list) in enumerate(region_list):
        for reg in reg_list:
            if not reg in deps['region'].values:
                print('Warning: ' + reg + ' not in region index.')
        index = [region in reg_list for region in deps['region'].values]
        patches[i] = deps.index[index].values
        sizes[i] = deps['population'][patches[i]].sum()
    return patches, sizes

