# -*- coding: utf-8 -*-
"""
Created on Sat May  5 20:01:07 2018

@author: Ilja
"""

from Orange.data import Table
from Orange.preprocess import Impute, Average

data = Table("heart_disease.tab")
imputer = Impute(method=Average())
impute_heart = imputer(data)

