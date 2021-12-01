import matplotlib.pyplot as plt
import torch
import numpy as np
import pdb
torchType = torch.float32


def plot_traj_coloured(z, weights, loglikelihood, x, transfo):
    
    dim = z.shape[-1]//2
    q = z[..., :dim]
    p = z[..., dim:]
    ####Contour
    x_lim= 3
    y_lim = 3 
    X = np.arange(-x_lim, x_lim, 0.05)
    n= len(X)
    Y = np.arange(-y_lim, y_lim, 0.05)
    X, Y = np.meshgrid(X, Y)
    #pdb.set_trace()
    z_tot = torch.cat([torch.tensor(X[...,None], dtype = torchType),torch.tensor(Y[...,None], dtype = torchType),
                      torch.zeros(n,n,dim-2)], dim=-1)
    likelihood_contour = loglikelihood(z_tot, x=x)
    
    
    
    ####pdb.set_trace()
    plt.rcParams["figure.figsize"] = (14, 6)
    x1 = q[:,:,0].numpy()
    y1 =q[:,:,1].numpy()
    
    px1 = p[:,:,0].numpy()
    py1 = p[:,:,1].numpy()
    
    loglikelihood_z = loglikelihood(q, x=x)
    contrib1 = (loglikelihood_z + weights.transpose(0,1)).numpy()

    f, ax =plt.subplots()
    # Create a set of line segments so that we can color them individually
    # This creates the points as a N x 1 x 2 array so that we can stack points
    # together easily to get the segments. The segments array for line collection
    # needs to be (numlines) x (points per line) x 2 (for x and y)
    points1 = np.array([x1, y1]).T.reshape(-1, 1, 2)
    #segments = np.concatenate([points[:-1], points[1:]], axis=1)
    colormap = plt.cm.get_cmap('coolwarm') #or any other colormap
    #norm = plt.Normalize(contrib.min(), contrib.max())
    size = 2
    CS = ax.contour(X, Y, likelihood_contour, levels = 15)
    ax.clabel(CS, inline=1, fontsize=1)
    Q = ax.quiver(x1, y1, px1, py1, contrib1,  cmap=colormap, scale = 1/0.01, width=0.0022, headlength=3.5, headaxislength=3.)
    #Q = ax.scatter(x1,y1,c=contrib1, s= size, cmap=colormap)#,  marker='*')
    #fig, axs = plt.subplots(2, 1, sharex=True, sharey=True)
    cb = f.colorbar(Q)
    cb.set_label('Weighted likelihood along trajectory')
    plt.xlim(-x_lim,x_lim)
    plt.ylim(-y_lim,y_lim)
    f.show()
    f.savefig('./pics/Gaussian_dim%i_gamma%.2f_h%.2f_K%i.pdf'%(dim, -1*transfo.gamma, transfo.dt,x1.shape[1]))