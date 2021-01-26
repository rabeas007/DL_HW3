import json
import torch
from jointvae.models import VAE


def load(path):

    path_to_specs = path + 'specs.json'
    path_to_model = path + 'model.pkl'

    # Open specs file
    with open(path_to_specs) as specs_file:
        specs = json.load(specs_file)

    # Unpack specs
    dataset = specs["dataset"]
    latent_spec = specs["latent_spec"]

    img_size = (3, 64, 64)

    # Get model
    model = VAE(img_size=img_size, latent_spec=latent_spec)
    model.load_state_dict(torch.load(path_to_model, map_location=lambda storage, loc: storage))

    return model
