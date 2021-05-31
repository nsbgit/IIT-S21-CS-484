# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:49:58 2018

@author: suhas
"""
import pandas as pd
import math

df = pd.read_csv('CustomerSurveyData.csv')
nTotal = len(df)

df[['CarOwnership', 'JobCategory']] = df[['CarOwnership','JobCategory']].fillna(value='Missing')


count = df['CreditCard'].value_counts()
gini_root = 1 - (((count[0]/nTotal)**2 + (count[1]/nTotal)**2 + (count[2]/nTotal)**2 + (count[3]/nTotal)**2 +(count[4]/nTotal)**2))



cross = pd.crosstab(index = df['CarOwnership'], columns = df['CreditCard'],margins = True, dropna = True)
crossTable = pd.crosstab(index = df['JobCategory'], columns = df['CreditCard'],margins = True, dropna = True)

def gini_calculate(crossTable,size):
 initial_value=1
 for z in range(0,(crossTable.columns.size-1)):
   sum=0  
   denom=0
  
   for x in range(0,len(size)): 
     
       sum+=crossTable.iloc[size[x],z]
       denom+=crossTable.iloc[size[x],-1]
   
   sqr=math.pow((sum/denom),2)
    
   initial_value-=sqr
 
 return initial_value 


def total_gini(crossTable,size):
    sum=0;
    for x in range(0,len(size)):
        sum+=crossTable.iloc[size[x],-1]
    sum1=sum/crossTable.iloc[-1,crossTable.columns.size-1]   
    return sum1;    


a=gini_calculate(cross,[0,1])
b=gini_calculate(cross,[2])
ans = total_gini(cross,[0,1])*a+ total_gini(cross,[2])*b
    
a=gini_calculate(cross,[0,2])
b=gini_calculate(cross,[1])
ans = total_gini(cross,[0,2])*a+ total_gini(cross,[1])*b

a=gini_calculate(cross,[1,2])
b=gini_calculate(cross,[0])
ans = total_gini(cross,[1,2])*a+ total_gini(cross,[0])*b

 
a = gini_calculate(crossTable,[0])
ax = gini_calculate(crossTable , [1,2,3,4,5,6])
f1 = total_gini(crossTable,[0]) * a + total_gini(crossTable, [1,2,3,4,5,6]) * ax

b = gini_calculate(crossTable,[1])
bx = gini_calculate(crossTable , [0,2,3,4,5,6])
f2 = total_gini(crossTable,[1]) * b + total_gini(crossTable, [0,2,3,4,5,6]) * bx

c = gini_calculate(crossTable,[2])
cx = gini_calculate(crossTable , [1,0,3,4,5,6])
f3 = total_gini(crossTable,[2]) * c + total_gini(crossTable, [1,0,3,4,5,6]) * cx

d = gini_calculate(crossTable,[3])
dx = gini_calculate(crossTable , [1,2,0,4,5,6])
f4 = total_gini(crossTable,[3]) * d + total_gini(crossTable, [1,2,0,4,5,6]) * dx

e = gini_calculate(crossTable,[4])
ex = gini_calculate(crossTable , [1,2,3,0,5,6])
f5 = total_gini(crossTable,[4]) * e + total_gini(crossTable, [1,2,3,0,5,6]) * ex

f = gini_calculate(crossTable,[5])
fx = gini_calculate(crossTable , [1,2,3,4,0,6])
f6 = total_gini(crossTable,[5]) * f + total_gini(crossTable, [1,2,3,4,0,6]) * fx

g = gini_calculate(crossTable,[6])
gx = gini_calculate(crossTable , [1,2,3,4,5,0])
f7 = total_gini(crossTable,[6]) * g + total_gini(crossTable, [1,2,3,4,5,0]) * gx






a = gini_calculate(crossTable,[0,1])
ax = gini_calculate(crossTable , [2,3,4,5,6])
f1 = total_gini(crossTable,[0,1]) * a + total_gini(crossTable, [2,3,4,5,6]) * ax

b = gini_calculate(crossTable,[0,2])
bx = gini_calculate(crossTable , [1,3,4,5,6])
f2 = total_gini(crossTable,[0,2]) * b + total_gini(crossTable, [1,3,4,5,6]) * bx

c = gini_calculate(crossTable,[0,3])
cx = gini_calculate(crossTable , [1,2,4,5,6])
f3 = total_gini(crossTable,[0,3]) * c + total_gini(crossTable, [1,2,4,5,6]) * cx

d = gini_calculate(crossTable,[0,4])
dx = gini_calculate(crossTable , [1,2,3,5,6])
f4 = total_gini(crossTable,[0,4]) * d + total_gini(crossTable, [1,2,3,5,6]) * dx

e = gini_calculate(crossTable,[0,5])
ex = gini_calculate(crossTable , [1,2,3,4,6])
f5 = total_gini(crossTable,[0,5]) * e + total_gini(crossTable, [1,2,3,4,6]) * ex

f = gini_calculate(crossTable,[0,6])
fx = gini_calculate(crossTable , [1,2,3,4,5])
f6 = total_gini(crossTable,[0,6]) * f + total_gini(crossTable, [1,2,3,4,5]) * fx


a = gini_calculate(crossTable,[1,2])
ax = gini_calculate(crossTable , [0,3,4,5,6])
f1 = total_gini(crossTable,[1,2]) * a + total_gini(crossTable, [0,3,4,5,6]) * ax

b = gini_calculate(crossTable,[1,3])
bx = gini_calculate(crossTable , [0,2,4,5,6])
f2 = total_gini(crossTable,[1,3]) * b + total_gini(crossTable, [0,2,4,5,6]) * bx

c = gini_calculate(crossTable,[1,4])
cx = gini_calculate(crossTable , [0,2,3,5,6])
f3 = total_gini(crossTable,[1,4]) * c + total_gini(crossTable, [0,2,3,5,6]) * cx

d = gini_calculate(crossTable,[1,5])
dx = gini_calculate(crossTable , [0,2,3,4,6])
f4 = total_gini(crossTable,[1,5]) * d + total_gini(crossTable, [0,2,3,5,6]) * dx

e = gini_calculate(crossTable,[1,6])
ex = gini_calculate(crossTable , [0,2,3,4,5])
f5 = total_gini(crossTable,[1,6]) * e + total_gini(crossTable, [0,2,3,4,5]) * ex




a = gini_calculate(crossTable,[2,3])
ax = gini_calculate(crossTable , [0,1,4,5,6])
f1 = total_gini(crossTable,[2,3]) * a + total_gini(crossTable, [0,1,4,5,6]) * ax

b = gini_calculate(crossTable,[2,4])
bx = gini_calculate(crossTable , [0,3,1,5,6])
f2 = total_gini(crossTable,[2,4]) * b + total_gini(crossTable, [0,3,1,5,6]) * bx

c = gini_calculate(crossTable,[2,5])
cx = gini_calculate(crossTable , [0,4,3,1,6])
f3 = total_gini(crossTable,[2,5]) * c + total_gini(crossTable, [0,4,3,1,6]) * cx

d = gini_calculate(crossTable,[2,6])
dx = gini_calculate(crossTable , [0,5,3,4,1])
f4 = total_gini(crossTable,[2,6]) * d + total_gini(crossTable, [0,5,3,4,1]) * dx




a = gini_calculate(crossTable,[3,4])
ax = gini_calculate(crossTable , [0,1,2,5,6])
f1 = total_gini(crossTable,[3,4]) * a + total_gini(crossTable, [0,1,2,5,6]) * ax

b = gini_calculate(crossTable,[3,5])
bx = gini_calculate(crossTable , [0,2,1,4,6])
f2 = total_gini(crossTable,[3,5]) * b + total_gini(crossTable, [0,2,1,4,6]) * bx

c = gini_calculate(crossTable,[3,6])
cx = gini_calculate(crossTable , [0,2,5,1,4])
f3 = total_gini(crossTable,[3,6]) * c + total_gini(crossTable, [0,2,5,1,4]) * cx





a = gini_calculate(crossTable,[4,5])
ax = gini_calculate(crossTable , [0,1,2,3,6])
f1 = total_gini(crossTable,[4,5]) * a + total_gini(crossTable, [0,1,2,3,6]) * ax

b = gini_calculate(crossTable,[4,6])
bx = gini_calculate(crossTable , [0,2,1,3,5])
f2 = total_gini(crossTable,[4,6]) * b + total_gini(crossTable, [0,2,1,3,5]) * bx


c = gini_calculate(crossTable,[5,6])
cx = gini_calculate(crossTable , [0,2,3,1,4])
f3 = total_gini(crossTable,[5,6]) * c + total_gini(crossTable, [0,2,3,1,4]) * cx
    



a = gini_calculate(crossTable,[0,1,2])
ax = gini_calculate(crossTable , [3,4,5,6])
f1 = total_gini(crossTable,[0,1,2]) * a + total_gini(crossTable, [3,4,5,6]) * ax

b = gini_calculate(crossTable,[0,1,3])
bx = gini_calculate(crossTable , [2,4,5,6])
f2 = total_gini(crossTable,[0,1,3]) * b + total_gini(crossTable, [2,4,5,6]) * bx

c = gini_calculate(crossTable,[0,1,4])
cx = gini_calculate(crossTable , [2,3,5,6])
f3 = total_gini(crossTable,[0,1,4]) * c + total_gini(crossTable, [2,3,5,6]) * cx

d = gini_calculate(crossTable,[0,1,5])
dx = gini_calculate(crossTable , [2,3,4,6])
f4 = total_gini(crossTable,[0,1,5]) * d + total_gini(crossTable, [2,3,4,6]) * dx

e = gini_calculate(crossTable,[0,1,6])
ex = gini_calculate(crossTable , [2,3,4,5])
f5 = total_gini(crossTable,[0,1,6]) * e + total_gini(crossTable, [2,3,4,5]) * ex



a = gini_calculate(crossTable,[0,2,3])
ax = gini_calculate(crossTable , [1,4,5,6])
f1 = total_gini(crossTable,[0,2,3]) * a + total_gini(crossTable, [1,4,5,6]) * ax

b = gini_calculate(crossTable,[0,2,4])
bx = gini_calculate(crossTable , [1,3,5,6])
f2 = total_gini(crossTable,[0,2,4]) * b + total_gini(crossTable, [1,3,5,6]) * bx

c = gini_calculate(crossTable,[0,2,5])
cx = gini_calculate(crossTable , [1,3,4,6])
f3 = total_gini(crossTable,[0,2,5]) * c + total_gini(crossTable, [1,3,4,6]) * cx

d = gini_calculate(crossTable,[0,2,6])
dx = gini_calculate(crossTable , [1,3,4,5])
f4 = total_gini(crossTable,[0,2,6]) * d + total_gini(crossTable, [1,3,4,5]) * dx





a = gini_calculate(crossTable,[0,3,4])
ax = gini_calculate(crossTable , [1,2,5,6])
f1 = total_gini(crossTable,[0,3,4]) * a + total_gini(crossTable, [1,2,5,6]) * ax

b = gini_calculate(crossTable,[0,3,5])
bx = gini_calculate(crossTable , [1,2,4,6])
f2 = total_gini(crossTable,[0,3,5]) * b + total_gini(crossTable, [1,2,4,6]) * bx

c = gini_calculate(crossTable,[0,3,6])
cx = gini_calculate(crossTable , [1,2,4,5])
f3 = total_gini(crossTable,[0,3,6]) * c + total_gini(crossTable, [1,2,4,5]) * cx








a = gini_calculate(crossTable,[0,4,5])
ax = gini_calculate(crossTable , [1,2,3,6])
f1 = total_gini(crossTable,[0,4,5]) * a + total_gini(crossTable, [1,2,3,6]) * ax

b = gini_calculate(crossTable,[0,4,6])
bx = gini_calculate(crossTable , [1,2,3,5])
f2 = total_gini(crossTable,[0,4,6]) * b + total_gini(crossTable, [1,2,3,5]) * bx






c = gini_calculate(crossTable,[0,5,6])
cx = gini_calculate(crossTable , [1,2,3,4])
f3 = total_gini(crossTable,[0,5,6]) * c + total_gini(crossTable, [1,2,3,4]) * cx




a = gini_calculate(crossTable,[1,2,3])
ax = gini_calculate(crossTable , [0,4,5,6])
f1 = total_gini(crossTable,[1,2,3]) * a + total_gini(crossTable, [0,4,5,6]) * ax

b = gini_calculate(crossTable,[1,2,4])
bx = gini_calculate(crossTable , [0,3,5,6])
f2 = total_gini(crossTable,[1,2,4]) * b + total_gini(crossTable, [0,3,5,6]) * bx

c = gini_calculate(crossTable,[1,2,5])
cx = gini_calculate(crossTable , [0,3,4,6])
f3 = total_gini(crossTable,[1,2,5]) * c + total_gini(crossTable, [0,3,4,6]) * cx

d = gini_calculate(crossTable,[1,2,6])
dx = gini_calculate(crossTable , [0,3,4,5])
f4 = total_gini(crossTable,[1,2,6]) * d + total_gini(crossTable, [0,3,4,5]) * dx




a = gini_calculate(crossTable,[1,3,4])
ax = gini_calculate(crossTable , [0,2,5,6])
f1 = total_gini(crossTable,[1,3,4]) * a + total_gini(crossTable, [0,2,5,6]) * ax

b = gini_calculate(crossTable,[1,3,5])
bx = gini_calculate(crossTable , [0,2,4,6])
f2 = total_gini(crossTable,[1,3,5]) * b + total_gini(crossTable, [0,2,4,6]) * bx

c = gini_calculate(crossTable,[1,3,6])
cx = gini_calculate(crossTable , [0,2,4,5])
f3 = total_gini(crossTable,[1,3,6]) * c + total_gini(crossTable, [0,2,4,5]) * cx



a = gini_calculate(crossTable,[1,4,5])
ax = gini_calculate(crossTable , [0,2,3,6])
f1 = total_gini(crossTable,[1,4,5]) * a + total_gini(crossTable, [0,2,3,6]) * ax

b = gini_calculate(crossTable,[1,4,6])
bx = gini_calculate(crossTable , [0,2,3,5])
f2 = total_gini(crossTable,[1,4,6]) * b + total_gini(crossTable, [0,2,3,5]) * bx




c = gini_calculate(crossTable,[1,5,6])
cx = gini_calculate(crossTable , [0,2,3,4])
f3 = total_gini(crossTable,[1,5,6]) * c + total_gini(crossTable, [0,2,3,4]) * cx





a = gini_calculate(crossTable,[2,3,4])
ax = gini_calculate(crossTable , [0,1,5,6])
f1 = total_gini(crossTable,[2,3,4]) * a + total_gini(crossTable, [0,1,5,6]) * ax

b = gini_calculate(crossTable,[2,3,5])
bx = gini_calculate(crossTable , [0,1,4,6])
f2 = total_gini(crossTable,[2,3,5]) * b + total_gini(crossTable, [0,1,4,6]) * bx

c = gini_calculate(crossTable,[2,3,6])
cx = gini_calculate(crossTable , [0,1,4,5])
f3 = total_gini(crossTable,[2,3,6]) * c + total_gini(crossTable, [0,1,4,5]) * cx



a = gini_calculate(crossTable,[2,4,5])
ax = gini_calculate(crossTable , [0,1,3,6])
f1 = total_gini(crossTable,[2,4,5]) * a + total_gini(crossTable, [0,1,3,6]) * ax

b = gini_calculate(crossTable,[2,4,6])
bx = gini_calculate(crossTable , [0,1,3,5])
f2 = total_gini(crossTable,[2,4,6]) * b + total_gini(crossTable, [0,1,3,5]) * bx




c = gini_calculate(crossTable,[2,5,6])
cx = gini_calculate(crossTable , [0,1,3,4])
f3 = total_gini(crossTable,[2,5,6]) * c + total_gini(crossTable, [0,1,3,4]) * cx



a = gini_calculate(crossTable,[3,4,5])
ax = gini_calculate(crossTable , [0,1,2,6])
f1 = total_gini(crossTable,[3,4,5]) * a + total_gini(crossTable, [0,1,2,6]) * ax

b = gini_calculate(crossTable,[3,4,6])
bx = gini_calculate(crossTable , [0,1,2,5])
f2 = total_gini(crossTable,[3,4,6]) * b + total_gini(crossTable, [0,1,2,5]) * bx




c = gini_calculate(crossTable,[3,5,6])
cx = gini_calculate(crossTable , [0,1,2,4])
f3 = total_gini(crossTable,[3,5,6]) * c + total_gini(crossTable, [0,1,2,4]) * cx



c = gini_calculate(crossTable,[4,5,6])
cx = gini_calculate(crossTable , [0,1,2,3])
f3 = total_gini(crossTable,[4,5,6]) * c + total_gini(crossTable, [0,1,2,3]) * cx
