import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
"""
dtype = torch.float
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)
"""
dtype = torch.float
#device = torch.device("cpu")
# Uncomment the line below to run on GPU
device = torch.device("cuda:0")

print(device)
# Standardize data
#x_train = np.array(train_dataset.data, dtype=np.float)

nb_inputs = 100
nb_hidden = 4
nb_outputs = 2

time_step = 1e-3
nb_steps = 200

batch_size = 256

freq = 5 # Hz
prob = freq*time_step  # 5e-3
mask = torch.rand((batch_size,nb_steps,nb_inputs), device=device, dtype=dtype) #256*200*100
x_data = torch.zeros((batch_size,nb_steps,nb_inputs), device=device, dtype=dtype, requires_grad=False)
x_data[mask<prob] = 1.0

print("x_data:",x_data.shape)
data_id = 0
plt.imshow(x_data[data_id].cpu().t(), cmap=plt.cm.gray_r, aspect="auto")
plt.xlabel("Time (ms)")
plt.ylabel("Unit")
plt.show()

# 教師ラベル
y_data = torch.tensor(1*(np.random.rand(batch_size)<0.5),device=device)
#print(y_data)

tau_mem = 10e-3
tau_syn = 5e-3

alpha = float(np.exp(-time_step/tau_syn))
beta = float(np.exp(-time_step/tau_mem))

weight_scale = 7*(1.0-beta) # this should give us some spikes to begin with

w1 = torch.empty((nb_inputs, nb_hidden),device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w1, mean=0.0, std=weight_scale/np.sqrt(nb_inputs))

w2 = torch.empty((nb_hidden, nb_outputs), device=device, dtype=dtype, requires_grad=True)
torch.nn.init.normal_(w2,mean=0.0,std=weight_scale/np.sqrt(nb_hidden))

print("init done")


h1 = torch.einsum("abc,cd->abd", (x_data, w1))

def spike_fn(x):
    out = torch.zeros_like(x)
    out[x>0]=1.0
    return out

syn = torch.zeros((batch_size,nb_hidden),device=device,dtype=dtype)
mem = torch.zeros((batch_size,nb_hidden), device=device, dtype=dtype)

# Here we define two lists which we use to record the membrane potentials and output spikes
mem_rec = [mem]
spk_rec = [mem]

for t in range(nb_steps):
    mthr = mem - 1.0
    out = spike_fn(mthr)
    rst = torch.zeros_like(mem)
    c = (mthr > 0)
    rst[c] = torch.ones_like(mem)[c]

    new_syn = alpha*syn + h1[:,t]
    new_mem = beta*mem + syn -rst

    mem = new_mem
    syn = new_syn

    mem_rec.append(mem)
    spk_rec.append(out)

# Now we merge the recorded membrane potentials into a single tensor
mem_rec = torch.stack(mem_rec,dim=1)
spk_rec = torch.stack(spk_rec,dim=1)

def plot_voltage_traces(mem, spk=None, dim=(3,5), spk_height=5):
    gs = GridSpec(*dim)
    if spk is not None:
        dat = (mem+spk_height*spk).detach().cpu().numpy()
        print("dat:",dat.shape)
    else:
        dat = mem.detach().cpu().numpy()
    for i in range(np.prod(dim)):
        if i==0: a0=ax=plt.subplot(gs[i])
        else:ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i])
        ax.axis("off")
    plt.show()
fig=plt.figure(dpi=100)
plot_voltage_traces(mem_rec, spk_rec)