import numpy as np
import matplotlib.bylab as plt
x = np.linspace(-2,2, 11) # -2 부터 2 까지 포인트를 만들겠다


x,y = np.meshgrid(x,y) #1차원 -> 2차원
print(x)

ax.axis('equal')