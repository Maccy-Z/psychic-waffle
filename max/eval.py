import numpy as np
import torch
from matplotlib import pyplot as plt

from ndp_main.data.datasets import SineData
from ndp_main.models.training import TimeNeuralProcessTrainer
from ndp_main.models.utils import context_target_split as cts


def get_model():
    basedir = "/home/maccyz/Documents/active_ode/ndp_main/main/results/1d/sine/ndp/ndp_sine"
    get_file = "trained_model.pth"
    with open(f'{basedir}/{get_file}', 'rb') as f:
        model = torch.load(f)
    return model

def get_sin_ds():
    x = torch.linspace(-3, 3, 100)
    y = 0.5 * torch.sin(x)
    return x, y


ds = SineData(amplitude_range=(-1., 1.), shift_range=(-.5, .5), num_samples=500)
x, y = get_sin_ds() # ds[0]
x, y, = x.reshape(1, -1, 1), y.reshape(1, -1, 1)
# Create context and target points and apply neural process
x_context, y_context, x_target, y_target, y_init = cts(x, y, 50, 0)
x_target = torch.sort(x_target, dim=1).values
nodep_model = get_model()
nodep_model.eval()

x_target = torch.arange(-3, 3, 0.1).reshape(1, -1, 1)
with torch.no_grad():
    preds = nodep_model(x_context, y_context, x_target, y_init)

means, std_devs = preds.loc, preds.scale
means = means.squeeze()
std_devs = std_devs.squeeze()

# Create a figure and a set of subplots
fig, ax = plt.subplots()

# Plot the means as a line
ax.plot(x_target.squeeze(), means, label='Mean')
# Shade the region between mean + std and mean - std
ax.fill_between(x_target.squeeze(), means - std_devs, means + std_devs, alpha=0.2)

# Plot out the original dataset
x_context, y_context = x_context.squeeze(), y_context.squeeze()
ax.plot(x_context, y_context, 'o', label='Context')

# Add some labels and a legend
ax.set_xlabel('Index')
ax.set_ylabel('Value')
ax.set_title('Means and Standard Deviations')
ax.legend()

# Show the plot
plt.show()