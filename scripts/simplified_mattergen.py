# Simplified MatterGen model that takes an input from a cif file 

from omegaconf import OmegaConf
from pymatgen.io.cif import CifParser
import numpy as np
import torch
from mattergen.common.data.dataset import CrystalDataset

#Importing Stuff from the actual diffusion_module, components for the diffusion part of the model
from mattergen.diffusion.diffusion_module import DiffusionModule
from mattergen.diffusion.corruption.multi_corruption import MultiCorruption
from mattergen.diffusion.corruption.sde_lib import SDE,VESDE,VPSDE
from mattergen.diffusion.corruption.d3pm_corruption import D3PMCorruption
from mattergen.diffusion.d3pm.d3pm import MaskDiffusion, create_discrete_diffusion_schedule
from mattergen.diffusion.losses import DenoisingScoreMatchingLoss
from mattergen.diffusion.score_models.base import ScoreModel
from mattergen.common.gemnet.gemnet import GemNetT
from mattergen.diffusion.timestep_samplers import UniformTimestepSampler
from mattergen.diffusion.data.batched_data import SimpleBatchedData
from mattergen.common.utils.data_utils import radius_graph_pbc

cfg = OmegaConf.load("mattergen/conf/data_module/custom_dataset.yaml")
cif_file = "mp-5229.cif"

parser = CifParser(cif_file)
structure = parser.parse_structures()[0]
print("Cif structure: ")
print(structure)

position = np.array(structure.frac_coords)
cell = np.array(structure.lattice.matrix)
atomic_numbers = np.array([site.specie.number for site in structure.sites])
number_atoms = np.array([len(structure.sites)])
structure_id = np.array([0])

dataset_simplified = CrystalDataset(
    pos=position,
    cell=cell,
    atomic_numbers=atomic_numbers,
    num_atoms=number_atoms,
    structure_id=structure_id,
    
)

data = {
    "positions": torch.tensor(position, dtype=torch.float32),
    "lattice": torch.tensor(cell, dtype=torch.float32),
    "atomic_types": torch.tensor(atomic_numbers, dtype=torch.long),
}

num_atoms = len(position)  # Number of atoms in the structure
batch_idx = {
    "positions": torch.zeros(num_atoms, dtype=torch.long),  # All atoms belong to batch 0
    "lattice": None,  # Lattice is dense, no batch index needed
    "atomic_types": torch.zeros(num_atoms, dtype=torch.long),  # All atoms belong to batch 0
}

# Create a SimpleBatchedData object
batched_dataset = SimpleBatchedData(data=data, batch_idx=batch_idx)

print("Simplified dataset: ")
print(dataset_simplified)


#Corruption for multiple parts
position_corruption = VPSDE(beta_min=0.1,beta_max=20)
lattice_corruption = VPSDE(beta_min=0.1,beta_max=20)

schedule = create_discrete_diffusion_schedule(
    kind="linear",
    beta_min=1e-3,
    beta_max=1e-1,
    num_steps=100,
    scale=1.0,
)

d3pm_process = MaskDiffusion(
    dim=128, 
    schedule=schedule, 
    precision=torch.float32,
    use_fast_inference=True
)
atom_corruption = D3PMCorruption(d3pm=d3pm_process)

corruption = MultiCorruption(
    sdes={"positions": position_corruption, "lattice": lattice_corruption},
    discrete_corruptions={"atomic_types": atom_corruption},
)

atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.long)

# Debugging: Check atomic numbers
print("Atomic numbers:", atomic_numbers)
print("Max atomic number in dataset:", atomic_numbers.max())
print("Min atomic number in dataset:", atomic_numbers.min())
print("Atomic numbers dtype:", atomic_numbers.dtype)

# Update AtomEmbedding if necessary
class AtomEmbedding(torch.nn.Module):
    def __init__(self, emb_size, max_atomic_number=128):
        super().__init__()
        self.emb_size = emb_size
        self.embedding = torch.nn.Embedding(max_atomic_number + 1, emb_size)
    def forward(self, atomic_numbers):
        print("Atomic types hitting atom_emb:", atomic_numbers)
        return self.embedding(atomic_numbers)


max_atomic_number = int(atomic_numbers.max())
atom_embedding = AtomEmbedding(emb_size=128, max_atomic_number=max_atomic_number)

lattice = torch.tensor(structure.lattice.matrix, dtype=torch.float32)
lattice = lattice.unsqueeze(0)
# Debugging: Print shape of lattice before passing to radius_graph_pbc
print("Lattice shape before radius_graph_pbc:", lattice.shape)

# Generate interaction graph
edge_index, to_jimages, num_bonds = radius_graph_pbc(
    cart_coords=torch.tensor(structure.cart_coords, dtype=torch.float32),
    lattice=lattice,
    num_atoms=torch.tensor([len(position)], dtype=torch.long),
    radius=6.0,
    max_num_neighbors_threshold=50,
    max_cell_images_per_dim=5,
)

score_model = GemNetT(
    num_targets=3,          
    latent_dim=128,
    atom_embedding=atom_embedding,
    num_spherical=7,
    num_radial=128,
    num_blocks=3,
    emb_size_atom=512,
    emb_size_edge=512,
    emb_size_trip=64,
    emb_size_rbf=16,
    emb_size_cbf=16,
    cutoff=6.0,
    activation="swish",
    max_neighbors=50,
)

#what the score will predict
model_targets = {
    "positions" : "score_times_std",
    "lattice": "score_times_std",
    "atomic_types": "logits",
}
#weights(lambda) for each loss in the loss function
weights = {
    "positions": 0.1,
    "lattice": 1.0,
    "atomic_types": 1.0,
}
#loss function with a concrete implementation
loss_fn = DenoisingScoreMatchingLoss(
    model_targets=model_targets,
    weights=weights,
)
diffusion_module = DiffusionModule(
    model=score_model,
    corruption=corruption,
    loss_fn=loss_fn,
    timestep_sampler=UniformTimestepSampler(min_t=1e-5, max_t=1.0),
)

# Corrupt the batch
noisy_batch, t = diffusion_module._corrupt_batch(batched_dataset)
print("Noisy batch at timestep:", t)


# Optional: Lattice matrix (if used by GemNetT)
lattice = noisy_batch["lattice"].unsqueeze(0)  # Add batch dimension if needed

z = None
# Extract required inputs from noisy_batch for GemNetT
frac_coords = noisy_batch["positions"]  # Fractional coordinates
atom_types = noisy_batch["atomic_types"]  # Atomic numbers/types
num_atoms = torch.tensor([len(frac_coords)])  # Number of atoms in each structure
batch = torch.zeros(len(frac_coords), dtype=torch.long).to(frac_coords.device)  # Batch indices

# Call score_fn with all required inputs
reconstructed_batch = diffusion_module.model(
    z=z,
    frac_coords=frac_coords,
    atom_types=atom_types,
    num_atoms=num_atoms,
    batch=batch,
    lattice=lattice,
    edge_index=edge_index,
    to_jimages=to_jimages,
    num_bonds=num_bonds,
)
print("Reconstructed batch:", reconstructed_batch)

# Calculate loss
loss, metrics = diffusion_module.calc_loss(batched_dataset)
print("Loss:", loss)
print("Metrics:", metrics)


