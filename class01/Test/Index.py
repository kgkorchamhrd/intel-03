import numpy as np


# c = no.array([[1,2,3],[-1,1,4]])

# print(LA.norm(c, axis=0))
# print(LA.norm(c, axis=1))
# print(LA.norm(c, ord=1, axis=1))
# print(LA.norm(c, ord=2, axis=1))

#========================================

input_image = np.expand_dims(
    resized_image.transpose(2, 0, 1), 0
)