import math
import numpy as np
import sys,os

#somma n primi numeri naturali
def somman(n):
   arr= np.arange(1,n+1,1)
   somma=0
   for i in range(len(arr)):
      somma=somma+arr[i]

   return somma   


def sommansqrt(n):
   arr= np.arange(1,n+1,1)
   sommquad=0
   for i in range(len(arr)):
      sommquad=sommquad+math.sqrt(arr[i])

   return sommquad   
#somma radici primi n numeri naturali      
