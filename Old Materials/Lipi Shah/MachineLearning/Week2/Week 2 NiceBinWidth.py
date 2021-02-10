# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 08:09:55 2018

@author: minlam
"""
import numpy as np

h = 5.5

u = np.log10(h)

v = np.sign(u) * np.ceil(np.abs(u))

nice_h = 10 ** v

print('The Nice Bin-Width = ', nice_h)