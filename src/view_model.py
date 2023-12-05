import torch
import torch.nn as nn
from torchviz import make_dot
from model import Generator


# Initialize an instance of your Generator
generator = Generator()

# Create a random input tensor to feed through the model (latent vector)
input_tensor = torch.randn(1, 100)

# Forward pass to generate the computational graph
output_tensor = generator(input_tensor)

# Visualize the computational graph
dot = make_dot(output_tensor, params=dict(generator.named_parameters()))
dot.render("generator_model", format="png")  