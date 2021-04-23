# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 09:42:31 2021

@author: Lenovo
"""
import numpy as np

f = open("test.txt", 'a')

array =np.ones((1,601))*25

f.write('\n')

for i in range(array.shape[1]):
    f.write(" "+str(array[0][i]))

f.write('\n')
f.close()

f = open('test.txt','r')
data = f.read()
out = data.split()
print(type(out))
print(out)