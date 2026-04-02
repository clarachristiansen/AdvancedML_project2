import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.func import vmap, jacfwd

class PLcurve:
    def __init__(self, x0, x1, N, device):
        """
        Represent the piecewise linear curve connecting x0 to x1 using
        a total of N nodes (including end-points)
        """
        super(PLcurve, self).__init__()
        self.x0 = x0.reshape(1, -1)  # 1xD
        self.x1 = x1.reshape(1, -1)  # 1xD
        self.N = N

        t = torch.linspace(0, 1, N, device=device).reshape(N, 1)  # Nx1
        c = (1 - t) @ self.x0 + t @ self.x1  # NxD
        self.params = c[1:-1]  # (N-2)xD
        self.params.requires_grad = True

    def points(self,device):
        c = torch.concatenate((self.x0, self.params, self.x1), axis=0).to(device)  # NxD
        return c

    def plot(self, name, device):
        c = self.points(device).detach().cpu().numpy()
        plt.plot(c[:, 0], c[:, 1], label=name)

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)

class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)
    
    def mean(self, x):
        out, _ = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return out

class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)
    def mean(self, z):
        if z.dim() == 1:              # single point: (M,)
            out = self.decoder_net(z.unsqueeze(0))   # (1, ...)
            return out.reshape(-1)                   # (784,)
        elif z.dim() == 2:            # batch: (B, M)
            out = self.decoder_net(z)                # (B, ...)
            return out.reshape(out.shape[0], -1)

class EnsembleVAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder_list, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(EnsembleVAE, self).__init__()
        self.prior = prior
        self.decoders = nn.ModuleList(decoder_list)
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()
        decoder_batch = int(z.shape[0] / 3)
        
        log_probs = []
        for i, decoder in enumerate(self.decoders):
            log_probs += [decoder(z[i*decoder_batch : (i+1)*decoder_batch,:]).log_prob(x[i*decoder_batch : (i+1)*decoder_batch, :])]
        log_probs = torch.concat(log_probs)
        elbo = torch.mean(
            log_probs - q.log_prob(z) + self.prior().log_prob(z) ##Mby split kl divergence
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)

def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break

### PART A funcitions
def decoder_jacobian(decoders, z):
    #z = z.detach().clone().requires_grad_(True)
    #return torch.autograd.functional.jacobian(decoder.mean, z) 
    def f_single(z):
        full_mean = []
        for decoder in decoders:
            full_mean += [decoder.mean(z)]
        return torch.mean(torch.stack(full_mean), axis=(0)) # shape (784,)

    # batched Jacobian: (P, 784, 2)
    J = vmap(jacfwd(f_single))(z)
    return J

def pullback_metric(decoders, z):
    J = decoder_jacobian(decoders, z)
    if z.dim() == 1:
        return J.T @ J                            # (2,2)
    return J.transpose(-1, -2) @ J               # (P,2,2)

def plot_metric(decoders, grid):
    X, Y = torch.meshgrid(grid, grid, indexing="ij")
    XY = torch.cat((X.reshape(-1, 1), Y.reshape(-1, 1)), dim=1) # (P,2)
    
    means = [decoder.mean(XY) for decoder in decoders]
    std_per_pixel = torch.std(torch.stack(means), dim=0) # (P,784)
    uncertainty = torch.mean(std_per_pixel, axis=1)

    plt.imshow(
        uncertainty.reshape(X.shape).detach().cpu().numpy().T,
        extent=(grid[0].item(), grid[-1].item(), grid[0].item(), grid[-1].item()),
        origin="lower"
    )
    plt.colorbar()
    plt.savefig('Project2/test.png')

def curve_energy(decoders, curve, device, monte_runs=20):
    num_decoder = len(decoders)
    energy = 0.0
    curve = curve.points(device)
    for i in range(len(curve) - 1):
        zi = curve[i:i+1]
        zj = curve[i+1:i+2]
        acc = 0.0
        for _ in range(monte_runs):
            l = torch.randint(0, num_decoder, (1,)).item()
            k = torch.randint(0, num_decoder, (1,)).item()

            fl = decoders[l].mean(zi)
            fk = decoders[k].mean(zj)

            acc += torch.sum((fl - fk)**2)
        energy += acc/monte_runs
    return energy

def connecting_geodesic(decoders, curve, device, steps=200, lr=1e-2):
    opt = optim.Adam([curve.params], lr=lr)

    for _ in range(steps):
        opt.zero_grad()
        loss = curve_energy(decoders, curve, device)
        loss.backward()
        opt.step()

    return curve



if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics", "PreCoV", "CoV", "PostCoV"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="Project2",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=60,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=80,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=100,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--model",
        type=int,
        default=0,
        metavar="M",
        help="What number of model is it?",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        metavar="S",
        help="What model to start with",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=9,
        metavar="E",
        help="What model to end with",
    )

    args = parser.parse_args()
    # print("# Options")
    # for key, value in sorted(vars(args).items()):
    #     print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net
    
    model = EnsembleVAE(
            GaussianPrior(M),
            [GaussianDecoder(new_decoder()), GaussianDecoder(new_decoder()), GaussianDecoder(new_decoder())],
            GaussianEncoder(new_encoder()),
        ).to(device)

    # Choose mode to run
    if args.mode == "train":
        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model_{args.model}.pt",
        )

    elif args.mode == "sample":
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model.load_state_dict(torch.load(args.experiment_folder + f"/model_{args.model}.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        experiments_folder = args.experiment_folder
        model.load_state_dict(torch.load(args.experiment_folder + f"/model_{args.model}.pt",weights_only=True))
        model.eval()

        dimss=7.5
        N = 100
        #plot_metric(G, torch.linspace(-r, r, 20, device=device))
        plot_metric(model.decoders, torch.linspace(-dimss, dimss, 100, device=device))
        #plot_metric_fast(model.decoder, torch.linspace(-r, r, 100))
        # test encoder mean
        all_z = []
        all_y = []

        for x, y in mnist_train_loader:
            x = x.to(device)
            z = model.encoder.mean(x) # torch.Size([32, 2])
            all_z.append(z.detach().cpu())
            all_y.append(y.cpu())

        z = torch.cat(all_z, dim=0)          # (N, 2)
        labels = torch.cat(all_y, dim=0)  # (N,)
        plt.scatter(z[:, 0], z[:, 1], s=1, c=labels)
        #plt.scatter(z[:, 0], z[:, 1], s=1) # without labels - better?
        #plt.show()
        plt.savefig(f'{experiments_folder}/test.png')
        ## HERE
        T = 20
        for _ in range(args.num_curves):
            idx = torch.randint(z.shape[0], (2,))
            c = PLcurve(z[idx[0]], z[idx[1]], T, args.device)
            e0 = curve_energy(model.decoders, c, args.device).item()
            print(f"Energy before optimization is {e0:.2f}")

            connecting_geodesic(model.decoders, c, args.device)
            
            e1 = curve_energy(model.decoders, c, args.device).item()
            print(f"Energy after optimization is {e1:.2f}")
            c.plot('After', args.device)
            print(f"drop: {(e0-e1)/max(e0,1e-12):.2%}")
            line = (torch.linspace(0, 1, T).unsqueeze(1) * c.x1 + (1-torch.linspace(0,1,T).unsqueeze(1)) * c.x0).to(device)
            dev = torch.norm(c.points(device).detach() - line.detach(), dim=1).max().item()
            print(f"max deviation: {dev:.3f}")

        plt.axis((-dimss, dimss, -dimss, dimss))
        plt.title('Latent Space: Uncertainty and Geodesics')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{experiments_folder}/latent_plot.png')
        plt.show()
    elif args.mode == "PreCoV":
        x_test, _ = next(iter(mnist_test_loader))
        num_pairs = args.num_curves
        indices = torch.randperm(len(x_test))[:2*num_pairs]
        pairs = [(indices[2*i], indices[2*i+1]) for i in range(num_pairs)]
        torch.save({"pairs": pairs, "x_test": x_test.cpu()}, "Project2/pairs.pt")
    elif args.mode =="CoV":
        data = torch.load("Project2/pairs.pt")
        pairs = data["pairs"]
        x_test = data["x_test"]
        num_pairs = args.num_curves

        num_decode = args.num_decoders

        for m in range(args.start, args.end+1):
            model.load_state_dict(torch.load(args.experiment_folder + f"/model_{m}.pt",weights_only=True))
            model.eval()

            for p_idx, (i,j) in enumerate(pairs):
                xi = x_test[i].unsqueeze(0).to(args.device)
                xj = x_test[j].unsqueeze(0).to(args.device)
                
                with torch.no_grad():
                    zi = model.encoder.mean(xi)
                    zj = model.encoder.mean(xj)

                d_euc = torch.norm(zi - zj).item()

                curve = PLcurve(zi.squeeze(), zj.squeeze(), N=20, device=args.device)
                connecting_geodesic(model.decoders[0:num_decode], curve, args.device)
                energy = curve_energy(model.decoders[0:num_decode], curve, args.device).item()
                d_geo = math.sqrt(energy)

                with open(f"Project2/{args.num_decoders}decoders{args.start}start{args.end}endTEST.txt", "a") as f:
                    f.write(f"{d_euc} \n")
                    f.write(f"{d_geo} \n")
                
    elif args.mode == "PostCoV":
        def load_file(path, results, num_decoders, num_pairs):
            with open(path, "r") as f:
                values = [float(x.strip()) for x in f.readlines()]
            for i in range(num_pairs):
                for j in range((len(values)//2)//num_pairs):
                    d_euc = values[2*i+num_pairs*j]
                    d_geo = values[2*i+num_pairs*j+1]
                    results[num_decoders]["euc"][i].append(d_euc)
                    results[num_decoders]["geo"][i].append(d_geo)

        #Load results
        num_pairs = args.num_curves
        results = {k : {
                "euc": [[] for _ in range(num_pairs)],
                "geo": [[] for _ in range(num_pairs)],}
                for k in [1,2,3]}

        for k in [1,2,3]:
            load_file(f"Project2/{k}decoders0start2end.txt", results, k, num_pairs)
            load_file(f"Project2/{k}decoders3start5end.txt", results, k, num_pairs)
            load_file(f"Project2/{k}decoders6start9end.txt", results, k, num_pairs)

        summ = {}

        for num_decode in results:
            cov_euc_list = []
            cov_geo_list = []
            
            for p_idx in range(num_pairs):
                euc_vals = torch.tensor(results[num_decode]["euc"][p_idx], device=args.device)
                geo_vals = torch.tensor(results[num_decode]["geo"][p_idx], device=args.device)

                cov_euc_list.append(torch.std(euc_vals) / torch.mean(euc_vals))
                cov_geo_list.append(torch.std(geo_vals) / torch.mean(geo_vals))
            
            cov_euc = torch.stack(cov_euc_list)
            cov_geo = torch.stack(cov_geo_list)

            summ[num_decode] = {"euc": torch.mean(cov_euc).item(),
                                "geo": torch.mean(cov_geo).item()}

        euc = [summ[k]["euc"] for k in [1,2,3]]
        geo = [summ[k]["geo"] for k in [1,2,3]]

        plt.figure()
        plt.plot([1,2,3],euc,label="Euclidean")
        plt.plot([1,2,3],geo,label="Geodesic")
        plt.xlabel("decoders")
        plt.ylabel("CoV")
        plt.title("CoV of distances vs number of decoders")
        plt.legend()
        plt.tight_layout()
        plt.savefig("Project2/CoV.png")
        plt.show()