import torch
import torchvision
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib.patches import FancyArrowPatch
from datetime import datetime

from pdb import set_trace

MIN_16, MAX_16 = torch.finfo(torch.float16).min, torch.finfo(torch.float16).max
MIN_32, MAX_32 = torch.finfo(torch.float32).min, torch.finfo(torch.float32).max

def stats(*args):
    for x in args:
        print("Type : ", type(x))
        print("Shape: ", x.shape)
        print("Sum  : ", x.sum())
        print("Mean : ", x.mean())
        print("STD  : ", x.std())
        print()
        

def torchCov(matrix:torch.Tensor, transposed=False, debug=False):
    "Transposed = True if individual samples are columns and not rows"
    if not isinstance(matrix, torch.Tensor): matrix = torch.tensor(matrix)
    if torch.cuda.is_available(): matrix = matrix.cuda()
    m = matrix.T if transposed else matrix
    if debug: set_trace()
    n = m.shape[0]
    MAX = torch.finfo(m.dtype).max
    mean = m.mean(axis=0, keepdim=True)
    m.sub_(mean)
    product = (m.T @ m).clamp(0, MAX)
    product[torch.isnan(product)] = 0
    product[torch.isinf(product)] = MAX
    return product / (n-1)

def torchPCA(matrix:torch.Tensor, k=2, transposed=False, fp16=True, debug=False):
    # Convert to tensor, cuda, half precision
    if debug: set_trace()
    dtype = torch.float16 if (fp16 and torch.cuda.is_available()) else torch.float32
    if not isinstance(matrix, torch.Tensor): matrix = torch.tensor(matrix).type(dtype)
    if torch.cuda.is_available(): 
        torch.cuda.set_device(0)
        matrix = matrix.cuda()
    # Make sure samples are rows and not columns
    m = matrix.T.type(dtype) if transposed else matrix.type(dtype)
    
    # PCA Computations
    now = datetime.now()
    cov_mat = torchCov(m, False, debug=debug).type(torch.float32)
    eig_vals, eig_vecs = cov_mat.eig(eigenvectors=True)
    eig_vals = eig_vals[:, 0] # Ignoring the complex part [:, 1]
    
    # Getting the top k eigen vectors
    order = eig_vals.argsort(descending=True)
    top_k = eig_vecs[:, order[:k]].type_as(m)

    # Reducing the matrix
    res = m @ top_k, top_k
    total_time = datetime.now() - now
    return res, total_time.microseconds / 1e6

# def torchCov(x, rowvar=False, bias=False, ddof=None, aweights=None):
#     """Estimates covariance matrix like numpy.cov"""
#     # ensure at least 2D
#     if x.dim() == 1: x = x.view(-1, 1)

#     # treat each column as a data point, each row as a variable
#     if rowvar and x.shape[0] != 1:
#         x = x.t()

#     if ddof is None:
#         if bias == 0: ddof = 1
#         else: ddof = 0

#     w = aweights
#     if w is not None:
#         if not torch.is_tensor(w): w = torch.tensor(w, dtype=torch.float)
#         w_sum = torch.sum(w)
#         avg = torch.sum(x * (w/w_sum)[:,None], 0)
#     else:
#         avg = torch.mean(x, 0)

#     # Determine the normalization
#     if w is None: fact = x.shape[0] - ddof
#     elif ddof == 0: fact = w_sum
#     elif aweights is None: fact = w_sum - ddof
#     else: fact = w_sum - ddof * torch.sum(w * w) / w_sum

#     xm = x.sub(avg.expand_as(x))

#     if w is None: X_T = xm.t()
#     else: X_T = torch.mm(torch.diag(w), xm).t()

#     c = torch.mm(X_T, xm)
#     c = c / fact

#     return c.squeeze()


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs
        
    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)
        
        
def visualize3dData(matrix:torch.Tensor, labels=None, transposed=False):
    if not isinstance(matrix, torch.Tensor): matrix = torch.tensor(matrix)
    m = matrix.clone().T if not transposed else matrix.clone()
    assert m.shape[0] == 3
    if labels is None:
        labels = torch.zeros(m.shape[1])
    else: 
        if not isinstance(labels, torch.Tensor): labels = torch.tensor(labels)
    
    fig = plt.figure(figsize=(16,16))
    ax = fig.add_subplot(111, projection='3d')
    plt.rcParams['legend.fontsize'] = 10
    
    classes = torch.unique(labels)
    for label in classes:
        data = m[:, labels == label]
        ax.plot(data[0, :], data[1, :], data[2, :],
                'o', markersize=8, alpha=0.4, label="Class 1")
    
    mean_vector = m.mean(dim=1, keepdim=True)
    cov_mat = torchCov(m, True)
    eig_vals, eig_vecs = cov_mat.eig(eigenvectors=True)
    eig_vals = eig_vals[:, 0] # Ignoring the complex part [:, 1]
    scaled_eig_vecs = (eig_vecs * eig_vals).cpu()
    
    means = mean_vector.cpu()
    for v in scaled_eig_vecs.T:
        a = Arrow3D([means[0].item(), v[0]], [means[1].item(), v[1]], [means[2].item(), v[2]],
                    mutation_scale=20, lw=3, arrowstyle="-|>", color="black")
        ax.add_artist(a)

    ax.set_xlabel('x_values')
    ax.set_ylabel('y_values')
    ax.set_zlabel('z_values')

    plt.title('Eigenvectors')

    plt.show()
    return fig
