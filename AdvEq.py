#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 16:19:11 2024

@author: arianna
"""

import numpy as np
import matplotlib.pyplot as plt
import advschemes

def main():
    "Solves the advection equation"
    w=advschemes.whiz()
    
    #calling the schemes
    w.ftbs()
    w.ftcs()
    w.ctcs()
    
    
    "Check on stability."
    "FTBS is stable and damping for 0<=c=1. It is stable and not damping for c=1. It is first order accurate."
    "CTCS is stable and not damping for 0<=c<=1. It is second order accurate."
    "FTCS is unconditionally unstable."


main()
                
