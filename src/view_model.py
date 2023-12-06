import torch
import torch.nn as nn
from model import Generator, Discriminator, Critic
D = 64
import numpy as np

# # Initialize an instance of your Generator
# generator = Generator()

# # Create a random input tensor to feed through the model (latent vector)
# input_tensor = torch.randn(1, 100)

# # Forward pass to generate the computational graph
# output_tensor = generator(input_tensor)

# # Visualize the computational graph
# dot = make_dot(output_tensor, params=dict(generator.named_parameters()))
# dot.render("generator_model", format="png")  
print("#########################################################")

dummy_input = torch.randn(1, 1, 512, 512)  # Eoutputample input size, adjust as needed
model = Critic()
output = model.initial(dummy_input)
print(output.shape)
output = model.resblock1(output)
print(output.shape)
output = model.resblock2(output)
print(output.shape)
output = model.resblock3(output)
print(output.shape)
output = model.resblock4(output)
print(output.shape)
output = model.flatten(output)
output = model.fc(output)
print(output.shape)
print("#########################################################")


model = Discriminator()
output = model.conv1(dummy_input)
output = model.leaky_relu(output)
print(output.shape)
output = model.conv2(output)
output = model.leaky_relu(output)
print(output.shape)
output = model.conv3(output)
output = model.leaky_relu(output)
print(output.shape)
output = model.conv4(output)
output = model.leaky_relu(output)
print(output.shape)
output = model.conv5(output)
output = model.leaky_relu(output)
print(output.shape)
output = model.flatten(output)
output = model.fc(output)
print(output.shape)
print("#########################################################")


noise = np.random.rand(1, 100).astype(np.float32)
noise = torch.from_numpy(noise)
model = Generator()
output = model.fc(noise)
output = model.unflatten(output)
output = model.relu(output)
output = model.upsample1(output)

output = model.conv1(output)
output = model.batch_norm1(output)
output = model.relu(output)
output = model.upsample1(output)

output = model.conv2(output)
output = model.batch_norm2(output)
output = model.relu(output)
output = model.upsample1(output)

output = model.conv3(output)
output = model.batch_norm3(output)
output = model.relu(output)
output = model.upsample1(output)

output = model.conv4(output)
output = model.batch_norm4(output)
output = model.relu(output)
output = model.upsample1(output)

output = model.conv5(output)
output = model.batch_norm5(output)
output = model.relu(output)
output = model.upsample1(output)

output = model.conv6(output)
output = model.batch_norm6(output)
output = model.relu(output)
output = model.upsample1(output)
output = model.conv7(output)
output = model.tanh(output)
print(output.shape)
print("#########################################################")
