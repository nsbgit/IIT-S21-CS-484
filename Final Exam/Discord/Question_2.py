import numpy as np

def CosineD(x, y):
   normX = np.sqrt(np.dot(x, x))
   normY = np.sqrt(np.dot(y, y))
   if (normX > 0.0 and normY > 0.0):
      outDistance = 1.0 - np.dot(x, y) / normX / normY
   else:
      outDistance = np.NaN
   return (outDistance)

# Question 2
# given focal point & points (0,0), (4,0), (0,4), (4,4)
# we will find the cosine distance between (2,2) and the rest of these points!

# (2,2) & (0,0)
x = np.array([2,2])
y = np.array([0,0])
print('Cosine distance btw (2,2) & (0,0) = ',CosineD(x, y))
# (2,2) & (4,0)
x = np.array([2,2])
y = np.array([4,0])
print('Cosine distance btw (2,2) & (4,0) = ',CosineD(x, y))
# (2,2) & (0,4)
x = np.array([2,2])
y = np.array([0,4])
print('Cosine distance btw (2,2) & (0,4) = ',CosineD(x, y))
# (2,2) & (4,4)
x = np.array([2,2])
y = np.array([4,4])
print('Cosine distance btw (2,2) & (4,4) = ',CosineD(x, y))
print('shortest distance is for (2,2) & (4,4)!')