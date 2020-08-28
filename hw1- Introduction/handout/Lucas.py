# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:04:39 2020

@author: KevinX
"""



def Lucas(n):
    if n == 1:
        return 2
    if n==2:
        return 1
    

    return Lucas(n - 1) + Lucas(n - 2)
    



    
    
