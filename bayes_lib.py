import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.models.gpytorch import GPyTorchModel
from IPython.display import clear_output
from botorch.generation.gen import gen_candidates_torch, get_best_candidates
from botorch.optim.initializers import gen_batch_initial_conditions
from botorch.fit import fit_gpytorch_model

class ExactGPModel(GPyTorchModel, gpytorch.models.ExactGP):
    _num_outputs = 1
    
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())        
        
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
    fit_gpytorch_model(mll)
    
#     for i in range(training_iter):
#         # Zero gradients from previous iteration
#         optimizer.zero_grad()
#         # Output from model
#         output = model(model.train_inputs[0])
#         # Calc loss and backprop gradients
#         loss = -mll(output, model._train_targets[0])
#         loss.backward()
#         if verbose:
#             print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
#                 i + 1, training_iter, loss.item(),
#                 model.covar_module.base_kernel.lengthscale.item(),
#                 model.likelihood.noise.item()
#             ))
#         optimizer.step()


def get_x_new(aquisition_func, model, best_f, device='cpu'):
    bounds = torch.tensor([[0., 1.]] * 6).T.to(device)
    Xinit = gen_batch_initial_conditions(aquisition_func, bounds, q=1, num_restarts=25, raw_samples=500)

    batch_candidates, batch_acq_values = gen_candidates_torch(
        initial_conditions=Xinit,
        acquisition_function=aquisition_func,
        lower_bounds=bounds[0],
        upper_bounds=bounds[1],
    )
    return get_best_candidates(batch_candidates, batch_acq_values)[0].detach()

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
        
        
def run_experiment(gt_func, n_iter, n_train_ini):
    best_fs = np.zeros(n_iter)
    # Initialize the experiment
    # ============================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_x = torch.rand(6*n_train_ini).reshape(-1,6).to(device)
    train_y = gt_func(train_x).to(device)
    best_f = train_y.max().item()
    for it in range(n_iter):
        # Run the forward model with hyperparam opt
        # ============================
        likelihood = GaussianLikelihood(noise_constraint=Interval(0.0,1e-4)).to(device)
        model = ExactGPModel(train_x, train_y, likelihood).to(device)
        train_hyper_params(model, likelihood)
        # New point aquisition
        # ============================
        EI = ExpectedImprovement(model, best_f=best_f, maximize=True)
        x_new = get_x_new(EI, model, best_f, device)
        y_new = gt_func(x_new).to(device)
        best_f = max(y_new.item(), best_f)

        train_x = torch.cat((train_x.reshape(-1,1), x_new.reshape(-1,1))).reshape(-1,6)
        train_y = torch.cat((train_y.reshape(-1,1), y_new.reshape(-1,1))).reshape(-1)
        # Print
        # ============================
        if (it+1)%print_period == 0:
            clear_output()
            print(f"{it+1:03d}/{n_iter}: {best_f:.5f}, {train_y[-1].item():.5f}, {np.log10(error_gap(best_f)):.5f}")

        # Record metric
        # ============================
        best_fs[it] = best_f
    return best_fs