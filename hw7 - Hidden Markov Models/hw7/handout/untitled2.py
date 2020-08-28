# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 12:20:18 2020

@author: KevinX
"""

test = [-123.37, -110.506, -101.0749,-95.4373]

train = [-132.6824, -116.31,-105.3125,-98.52]

seq = [10,100,1000,10000]

import matplotlib.pyplot as plt 

plt.plot(seq,test)
plt.plot(seq,train)
plt.ylabel('average log likelihood')
plt.xlabel('sequence')
plt.legend(('train', 'test'))