import numpy as np
import torch
from argparse import ArgumentParser

from utils.data import make_dataloaders, save_model, DotDict, str2bool
from models import VAE


def estimate_on_valset(model, val_loader, n_samples = 50):
    estimates = []
    for batch in val_loader:
        estimates.append(model.importance_estimate(n_samples, batch).detach().cpu().numpy())
    return -np.hstack(estimates).mean()


def train(args):
    '''
    Main training loop & metrics computation
    '''
    if args.device == "gpu":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"
    print("using device "+device.type)
    train_loader, val_loader = make_dataloaders(args.dataset, args.batch_size, device, subset = args.subset, flatten = args.flatten, binarize = args.binarize)
    print(args.dataset + " is loaded")


    model = None
    if args.model == "vae":
        model = VAE.VanillaVAE(latent_dim = args.latent_dim, archi=args.archi)
    elif args.model == "iwae":
        model = VAE.IWAE(num_samples= args.num_samples, latent_dim = args.latent_dim, archi=args.archi)
    elif args.model == "flowvae":
        model = VAE.FlowVAE(num_flows = args.num_flows, latent_dim = args.latent_dim, archi=args.archi)
    elif args.model == "neqvae":
        transformation_params = DotDict()
        transformation_params.h = 0.1
        transformation_params.gamma = -0.1
        transformation_params.name = 'DampedHamiltonian' # so far implemented : 'LeapFrog', 'DampedHamiltonian', 'Identity'
        a = torch.Tensor([1.,1.] )
        logvar_p = torch.tensor([0.]*args.latent_dim)
        if args.logvar_p != None:
            logvar_p = args.logvar_p
        model = VAE.NeqVAE(transformation_params=transformation_params, a=a, latent_dim = args.latent_dim, archi=args.archi, logvar_p=logvar_p)
    elif args.model == "neqvae2":
        transformation_params = DotDict()
        transformation_params.h = 0.1
        transformation_params.gamma = -0.1
        transformation_params.name = 'DampedHamiltonian_lf' # so far implemented : 'LeapFrog', 'DampedHamiltonian', 'Identity'
        a = torch.Tensor([1.,1.,1.] )
        model = VAE.NeqVAE(transformation_params=transformation_params, a=a, latent_dim = args.latent_dim, archi=args.archi)
    print(args.model +" "+ args.archi + " training begins...")
    losses, val_losses = [], []
    model.to(device)
    optim = model.get_optimizer()

    # train loop
    for epoch in range(args.epochs):
        print(f"epoch {epoch}")
        for batch in train_loader:
            optim.zero_grad()
            loss, x_hat, z, BCE = model.step(batch)
            losses.append(loss.item())
            loss.backward()
            optim.step()
            if args.model=="neqvae":
                model.zero_grad()

    # val loss
    for batch in val_loader:
        loss, _, _, _ = model.step(batch)
        val_losses.append(loss.item())

    nll = estimate_on_valset(model, val_loader, 50)

    save_model(model, args.model+"_"+args.dataset+".ckpt")
    return np.mean(val_losses), nll



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--latent_dim", default=2, type=int)
    parser.add_argument("--num_samples", default=5, type=int)
    parser.add_argument("--num_flows", default=2, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--subset", default=None, type=int)
    parser.add_argument("--flatten", default=True, type=str2bool)
    parser.add_argument("--binarize", default=True, type=str2bool)
    parser.add_argument("--logvar_p", default=None, choices=[None, 'adaptative'])
    parser.add_argument("--device", default="gpu", choices=['cpu', 'gpu'])
    parser.add_argument("--model", default="vae",  choices=['vae', 'iwae', 'flowvae', 'neqvae', 'neqvae2'])
    parser.add_argument("--archi", default="basic",  choices=['basic', 'large', 'convMnist', 'convCifar'])
    parser.add_argument("--dataset", default='fashionmnist', choices=['mnist', 'fashionmnist', 'cifar'])

    args = parser.parse_args()
    val_loss, elbo_est = train(args)
    print(f"train successful! Validation loss: {val_loss:.2f} elbo estimate {elbo_est:.2f}")
