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
    def __init__(self, train_x, train_y, n_steps):
        super(GPModule, self).__init__()
        # create data variables
        self.n_steps = n_steps
        self.length = train_x.shape[0]
        self._X = torch.zeros(self.n_steps, train_x.shape[-1])
        self._y = torch.zeros(self.n_steps)

        self._X[:self.length] = train_x
        self._y[:self.length] = train_y

        '''
        # initialise the model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModel(self._X, self._y, self.likelihood).double()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        '''
        # training steps
        self.training_iter = 50 



    def fit(self, data_x, data_y):
        """Train the GP
        """
        self._X[self.length : self.length+len(data_x)] = data_x
        self._y[self.length : self.length+len(data_y)] = data_y
        self.length += len(data_x)

        train_x = self._X[:self.length]
        train_y = self._y[:self.length]

        self.model = ExactGPModel(train_x, train_y, self.likelihood).double()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)


        self.model.train()
        self.likelihood.train()
        for i in range(self.training_iter):
            self.optimizer.zero_grad()
            output = self.model(train_x)
            loss = - self.mll(output, train_y)
            loss.backward()

            self.optimizer.step()
        

    def predict(self, state):
        """Predict the observable given a state
        """
        self.model.eval()
        self.likelihood.eval()

        observed_pred = self.likelihood(self.model(state))
        lower, upper = observed_pred.confidence_region()
        std = (upper - lower) / 2.
        
        return observed_pred.loc.detach(), std.detach()
