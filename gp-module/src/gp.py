import gpytorch
import torch

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GPModule:
    def __init__(self, n_steps=500):
        super(GPModule, self).__init__()
        # create data variables
        self.n_steps = n_steps
        self.length = 0
        self._X = torch.zeros(self.n_steps, 2)
        self._y = torch.zeros(self.n_steps)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        # training steps
        self.training_iter = 100


    def fit(self, data_x, data_y):
        """Train the GP
        """
        # update the data
        self._X[self.length : self.length+len(data_x)] = data_x
        self._y[self.length : self.length+len(data_y)] = data_y
        self.length += len(data_x)

        train_x = self._X[:self.length]
        train_y = self._y[:self.length]

        # initialise the model
        self.model = ExactGPModel(train_x, train_y, self.likelihood).double()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.model.train()
        self.likelihood.train()
        for i in range(self.training_iter):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = - self.mll(output, train_y)
            loss.backward()

            self.optimizer.step()
        #print(f'loss: {loss}')
        

    def predict(self, state):
        """Predict the observable given a state
        """
        self.model.eval()
        self.likelihood.eval()

        prediction = self.likelihood(self.model(state))
        return prediction.mean.detach(), prediction.stddev.detach()
