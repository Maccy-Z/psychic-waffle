import numpy as np
import matplotlib.pyplot as plt

print(np.__version__)

# Generate a 2D array (10x10) of random numbers
data = np.random.rand(10, 10)

# Plot the 2D array using matplotlib's imshow
plt.imshow(data, cmap='viridis', interpolation='none')
plt.colorbar()  # Add a colorbar to show the scale
plt.title("2D Array of Random Numbers")
plt.show()
