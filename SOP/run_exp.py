from pca_utils import *
from sklearn.datasets import make_classification
import gc
from tqdm import tqdm
import pandas as pd

from pdb import set_trace

full, half = torch.float32, torch.float16

def get_gaussian_sampler(dimensions=5, mean=0.0, variance=1.0, var_factor=0.25):
    mu = torch.tensor([mean]*dimensions).float()
    cov= torch.eye(dimensions).float() * (variance**2) * var_factor
    sampler = torch.distributions.MultivariateNormal(mu, cov)
    return sampler

def scale_to_01(x):
    return x.sub(x.min()).div(x.max())

def absolute_deviation(preds, targets):
    return (preds - targets).mean().abs()

eps = 1e-7
def relative_deviation(preds, targets):
    return ((preds - targets) / (targets + eps)).mean().abs()

def process_results(x32, eigs32, time32, x16, eigs16, time16):
    dict1 = {}
    dict1['abs_deviation'], dict1['rel_deviation'], dict1['time'] = {}, {}, {}
    dict1['abs_deviation']['data'] = absolute_deviation(x16, x32).item() if type(x16) != int else 0
    dict1['rel_deviation']['data'] = relative_deviation(x16, x32).item() if type(x16) != int else 0
    dict1['abs_deviation']['eigv'] = absolute_deviation(eigs16, eigs32).item() if type(eigs16) != int else 0
    dict1['rel_deviation']['eigv'] = relative_deviation(eigs16, eigs32).item() if type(eigs16) != int else 0
    dict1['time'][16] = time16
    dict1['time'][32] = time32
    return dict1

def get_results(x, iterations=10, scaled_only=True, debug=False):
    res = {}
    time = torch.zeros((2, 50))
    if x.shape[0] >= 1000000:
        if x.shape[0] >= 5000000: iterations = 2
        else: iterations = 5
    if not scaled_only:
        if debug: print("Processing non-scaled dataset")
        for i in range(iterations):
            (x32, eigs32), time[0, i] = torchPCA(x, k=3, fp16=False)
            (x16, eigs16), time[1, i] = torchPCA(x, k=3, fp16=True)
        time32, time16 = time[0,:].mean().item(), time[1,:].mean().item()
    else:
        x32, eigs32, time32, x16, eigs16, time16 = 0, 0, 0, 0, 0, 0        
    
    res['non_scaled'] = process_results(x32, eigs32, time32, x16, eigs16, time16)
    
    x = scale_to_01(x)
    
    time = torch.zeros((2, 50))
    if debug: print("Processing scaled (to 0-1 range) dataset")
    for i in range(iterations):
        (x32, eigs32), time[0, i] = torchPCA(x, k=3, fp16=False)
        (x16, eigs16), time[1, i] = torchPCA(x, k=3, fp16=True)
    time32, time16 = time[0,:].mean().item(), time[1,:].mean().item()
    
    res['scaled'] = process_results(x32, eigs32, time32, x16, eigs16, time16)
    
    return res


def write_result(i:int, n:int, dimensions:int, k:int, mean:float, var:float, res:dict, df:pd.DataFrame,
                 output:str="results.csv"):
    scaled, nscaled = res['scaled'], res['non_scaled']
    absd, reld = 'abs_deviation', 'rel_deviation'
    res1 = [scaled[absd]['data'], scaled[absd]['eigv'], scaled[reld]['data'], scaled[reld]['eigv']]
    res1+= [scaled['time'][16], scaled['time'][32]]
    res2 = [nscaled[absd]['data'], nscaled[absd]['eigv'], nscaled[reld]['data'], nscaled[reld]['eigv']]
    res2+= [nscaled['time'][16], nscaled['time'][32]]
    res1 = [True, n, dimensions, k, mean, var] + res1
    res2 = [False, n, dimensions,k, mean, var] + res2
    df.iloc[i]   = res1
    df.iloc[i+1] = res2
    if i % 5 == 0: 
        df.to_csv(output)
        print("Writing to CSV: done!")
        
mean_vars = [[0, 1], [0, 4], [0, 16], [0, 64], [0, 128], [0, 512], [0, 2048], [0, 8192], [0, 32768], [0, 65519]]
mean_vars+= [[1.5, 0.5], [1.5, 3], [3, 1]]
mean_vars+= [[6, 2], [6, 12], [12, 4]]
mean_vars+= [[24, 8], [24, 48], [48, 16]]
mean_vars+= [[96, 32], [96, 192], [192, 64]]
mean_vars+= [[384, 128], [384, 768], [768, 256]]
mean_vars+= [[1536, 512], [1536, 3072], [3072, 1024]]
mean_vars+= [[6144, 2048], [6144, 12288], [12288, 4096]]
mean_vars+= [[24576, 8192], [24576, 32768], [49152, 16384]]

n_values  = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000
            ,5000000, 10000000]
dimensions= [10, 50, 100, 200, 400]
# dimensions= [10,]


total = len(n_values) * len(mean_vars) * len(dimensions) * 2
k = 3
max_retries = 8
print(total)

df = pd.DataFrame(np.zeros((total, 12)), columns=['Scaled', 'N', 'Dimensions', 'K', 'Mean', 'Variance',
                                                'abs_deviation_data', 'abs_deviation_eigv',
                                                'rel_deviation_data', 'rel_deviation_eigv',
                                                'time_16', 'time_32'])

df.Scaled = df.Scaled.astype(bool)
df.N = df.N.astype(int)
df.Dimensions = df.Dimensions.astype(int)
df.K = df.K.astype(int)


skipped = []
pos = 0
debug = False
for dimension in dimensions:
    print(f"Number of columns: {dimension}")
    for i, (mean, var) in tqdm(list(enumerate(mean_vars))):
        if debug and i == 2: break
        sampler = get_gaussian_sampler(dimension, mean, var)
        df.to_csv("results.csv")
        for n in tqdm(n_values):
            if dimension == 400 and n == 10000000: continue
            still_doing = max_retries
            scaled_only=False
            while still_doing > 0:
                try:
                    data = sampler.sample((n,))
                    res = get_results(data, scaled_only=scaled_only)
                    still_doing = 0
                    write_result(pos, n, dimension, k, mean, var, res, df)
                    pos += 2
                except:
                    still_doing -= 1
                    if still_doing == max_retries // 2:
                        scaled_only = True
                    if still_doing == 0:
                        print(f"Skipping N={n}\t Mean={mean}\t Var={var}")
                        skipped.append([n, mean, var])
                        
df.to_csv("results.csv")
skip_f = open("skipped.txt", "w")
skip_f.write(str(skipped) + "\n")
skip_f.close()
