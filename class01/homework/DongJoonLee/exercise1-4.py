from PIL import Image
import numpy as np

image_path = './image.png'
image = Image.open(image_path)

image_array = np.array(image)

image_with_batch = np.expand_dims(image_array, axis=0) 

transposed_image = image_with_batch.transpose(0, 3, 1, 2) 

# Verify the shape of the result
print("Original Shape:", image_array.shape)
print("Shape after adding batch dimension:", image_with_batch.shape)
print("Shape after transposing:", transposed_image.shape)