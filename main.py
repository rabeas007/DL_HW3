import json

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn import manifold

from jointvae.models import VAE
from jointvae.training import Trainer
from torch import optim
import torch
from viz.visualize import Visualizer as Viz
from utils.load_model import load
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as dsets
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.decomposition import PCA
import seaborn as sns

batch_size = 256
learning_rate = 5e-4
epochs = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    # Check for cuda
    use_cuda = torch.cuda.is_available()

    # Load data
    train_transformation = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = dsets.ImageFolder('/datashare', train_transformation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define latent spec and model
    img_size = (3, 64, 64)
    latent_spec = {'cont': 32, 'disc': [10, 5, 5, 1]}
    model = VAE(img_size=img_size, latent_spec=latent_spec, use_cuda=use_cuda)
    if use_cuda:
        model.cuda()

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define trainer
    trainer = Trainer(model, optimizer,
                      cont_capacity=[0.0, 5.0, 25000, 30],
                      disc_capacity=[0.0, 5.0, 25000, 30],
                      use_cuda=use_cuda)

    # Train model for 100 epochs
    viz = Viz(model)
    training_cost, training_kl_loss = trainer.train(train_loader, epochs, ('img.gif', viz))

    # Plot
    f = open("NN_results", "w")
    json.dump([training_kl_loss, training_cost], f)
    f.close()

    make_plots("NN_results", epochs)

    # Load model
    # model = load("trained_models/celeba/")

    samples = viz.samples()

    plot_latent(model, train_loader, batch_size)
    # Save trained model
    torch.save(trainer.model.state_dict(), 'model.pkl')


def plot_training_cost(kl_loss, training_cost, num_epochs, training_cost_xmin):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(training_cost_xmin, num_epochs),
            training_cost[training_cost_xmin:num_epochs],
            color='#2A6EA6', label='Reconstruction Loss')

    ax.plot(np.arange(training_cost_xmin, num_epochs),
            kl_loss[training_cost_xmin:num_epochs],
            color='#FFA933', label='KL divergence')

    ax.grid(True)
    ax.set_xlim([training_cost_xmin, num_epochs])
    ax.set_xlabel('Epoch')
    plt.legend(loc="lower right")
    plt.show()


def make_plots(filename, num_epochs, training_cost_xmin=0):
    f = open(filename, "r")
    kl_loss, training_cost = json.load(f)
    f.close()

    plot_training_cost(kl_loss, training_cost, num_epochs, training_cost_xmin)


def plot_latent(autoencoder, data, batch_size):
    for i, (x, y) in enumerate(data):
        latent_dist = autoencoder.encode(x.to(device))
        z = autoencoder.reparameterize(latent_dist)
        z = z.to('cpu').detach().numpy()

        if i > batch_size:
            print("Computing t-SNE embedding...")
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
            X_tsne = tsne.fit_transform(z)

            # Plot images according to t-sne embedding
            print("Plotting t-SNE visualization...")
            fig, ax = plt.subplots()
            imscatter(X_tsne[:, 0], X_tsne[:, 1], imageData=x, ax=ax, zoom=0.6)
            plt.savefig('img1.png')

            fig, ax = plt.subplots()
            ax.scatter(*X_tsne.T)
            ax.set_xlabel('$c_1$')
            ax.set_ylabel('$c_2$')

            plt.savefig('img2.png')
            break

            # pca = PCA(2)
            # projection = pca.fit_transform(z)
            #
            # plt.scatter(*projection.T, c=np.argmax(y.numpy()), cmap="Set1")
            # plt.show()


def imscatter(x, y, ax, imageData, zoom):
    images = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        img = imageData[i] * 255.
        # img = img.astype(np.uint8).reshape([28, 28])
        img = img.resize_((28, 28))
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        image = OffsetImage(img, zoom=zoom)
        ab = AnnotationBbox(image, (x0, y0), xycoords='data', frameon=False)
        images.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()


if __name__ == '__main__':
    main()
