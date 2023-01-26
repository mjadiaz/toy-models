import gpytorch
import torch
from gp import ExactGPModel
import math
from tqdm import tqdm
import matplotlib.pyplot as plt

import numpy as np
from toy_models.envs.toy_functions import Simulator, TF2D_DEFAULT_CONFIG, minmax
config = TF2D_DEFAULT_CONFIG
simulator = Simulator(config)
x = np.linspace(10, 15, 10)
y = np.linspace(10, 15, 10)
xx, yy = np.meshgrid(x, y)
X = np.random.uniform(10, 15, 100)#xx.ravel()
Y = np.random.uniform(10, 15, 100)#yy.ravel()
Z = simulator.run(X, Y)
train_x = torch.tensor(np.concatenate((X[..., None], Y[..., None]), axis=1))
train_y = torch.tensor(Z)
print('Generated data')

#train_x = torch.linspace(0, 1, 100)
#train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood).double()

model.train()
likelihood.train()
print('Initialised the model - start training')
# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50
for i in tqdm(range(training_iter)):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    #print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
    #    i + 1, training_iter, loss.item(),
    #    model.covar_module.base_kernel.lengthscale.item(),
    #    model.likelihood.noise.item()
    #))
    optimizer.step()


# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    x = np.linspace(10, 15, 100)
    y = np.linspace(10, 15, 100)
    xx, yy = np.meshgrid(x, y)
    _X = xx.ravel()
    _Y = yy.ravel()
    Z = simulator.run(_X, _Y)
    test_x = torch.tensor(np.concatenate((_X[..., None], _Y[..., None]), axis=1)).double()
    observed_pred = likelihood(model(test_x))
    print('predicted')
    # Initialize plot
    f, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax.flatten()
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    breakpoint()
    # Plot training data as black stars
    #ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line

    img0 = ax[0].scatter(test_x.numpy()[:, 0], test_x.numpy()[:, 1], c=10*observed_pred.mean.numpy())
    ax[0].scatter(X, Y, c='white', s=10)
    f.colorbar(img0, ax=ax[0])
    # Shade between the lower and upper confidence bounds
    #ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    #the std is not correct, need to understand how to extrapolate
    img1 = ax[1].scatter(test_x.numpy()[:, 0], test_x.numpy()[:, 1], c=(upper.numpy()-lower.numpy())/2)
    f.colorbar(img1, ax=ax[1])
    #ax.set_ylim([-3, 3])
    #ax.legend(['Observed Data', 'Mean', 'Confidence'])
plt.show()
