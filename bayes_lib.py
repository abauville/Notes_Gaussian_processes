import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models.gpytorch import GPyTorchModel
from IPython.display import clear_output

class ExactGPModel(GPyTorchModel, gpytorch.models.ExactGP):
    _num_outputs = 1
    
    def __init__(self, train_x, train_y, likelihood):
        # super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        super().__init__(train_x.reshape(-1), train_y.reshape(-1), likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.covar_module = gpytorch.kernels.RBFKernel()
        # self.covar_module.lengthscale = 0.4
        self.to(train_x)
        
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
   

def train_hyper_params(model, likelihood, training_iter=20, verbose=False):
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.base_kernel.lengthscale.item(),
                model.likelihood.noise.item()
            ))
        optimizer.step()
        
def run_bayes_opt(Model, likelihood, train_x, train_y, gt_function, error_function=None, num_test_points=100, min_iter=2, max_iter=100):
    """
Parameters
===========
Model: class that inherits from Botorch model, e.g. GPyTorchModel, (and optionally one of gpytorch.models)
train_x: torch.tensor
    coordinates of the initial training points (i.e. observations)
train_y: torch.tensor
    target value of the training points
num_test_points: int
    number of test points (identical for every dimension)
gt_function: class
    a class that defines the test bounds, a global minimum and a method f that takes train_x, train_y as input
    See test_functions.py for examples
error_function: function, optional
    a function that computes a scalar error between the model and the ground truth
"""
    min_iter = max(min_iter, 2); 
    it = 0
    while (it<min_iter or (train_x[-1] - train_x[-2]).abs()>1e-6) and it<max_iter:
        # Run the forward model with hyperparam opt
        model = Model(train_x, train_y, likelihood)
        train_hyper_params(model, likelihood)
        
        # Run the aquisition method
        EI = ExpectedImprovement(model, best_f=0.2)
        ei = EI(test_x.reshape(-1,1,1))
        
        # Add the suggested point to the training points
        x_new = test_x[ei.argmax().item()]
        train_x = torch.cat((train_x.reshape(-1,1), x_new.reshape(-1,1))).reshape(-1)
        train_y = torch.cat((train_y.reshape(-1,1), f(x_new).reshape(-1,1))).reshape(-1)
        it += 1
        
        if error_function:
            error_function()
        
        