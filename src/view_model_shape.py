import torch
from model import Generator, Discriminator, ResnetDiscriminator, ResnetGenerator, UNetGenerator
D = 64
import numpy as np

print("#########################################################")

dummy_input = torch.randn(1, 1, 512, 512)  # Eoutputample input size, adjust as needed
model = ResnetDiscriminator()
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
print(output.shape)


output = model.conv1(output)
output = model.batch_norm1(output)
output = model.relu(output)
output = model.upsample1(output)
print(output.shape)

output = model.conv2(output)
output = model.batch_norm2(output)
output = model.relu(output)
output = model.upsample1(output)
print(output.shape)

output = model.conv3(output)
output = model.batch_norm3(output)
output = model.relu(output)
output = model.upsample1(output)
print(output.shape)

output = model.conv4(output)
output = model.batch_norm4(output)
output = model.relu(output)
output = model.upsample1(output)
print(output.shape)

output = model.conv5(output)
output = model.batch_norm5(output)
output = model.relu(output)
output = model.upsample1(output)
print(output.shape)

output = model.conv6(output)
output = model.batch_norm6(output)
output = model.relu(output)
output = model.upsample1(output)
print(output.shape)

output = model.conv7(output)
output = model.tanh(output)
print(output.shape)
print("#########################################################")
model = ResnetGenerator()
output = model.fc(noise)
output = model.unflatten(output)
output = model.relu(output)
output = model.upsample(output)
print(output.shape)

output = model.resblock1(output)
output = model.upsample(output)
print(output.shape)

output = model.resblock2(output)
output = model.upsample(output)
print(output.shape)

output = model.resblock3(output)
output = model.upsample(output)
print(output.shape)

output = model.resblock4(output)
output = model.upsample(output)
print(output.shape)

output = model.resblock5(output)
output = model.upsample(output)
print(output.shape)

output = model.resblock6(output)
output = model.upsample(output)
print(output.shape)

output = model.resblock7(output)
output = model.tanh(output)
print(output.shape)

print("#########################################################")
model = UNetGenerator()
down1 = model.down1(noise)
down2 = model.down2(down1)
down3 = model.down3(down2)
# Bottom
bottom = model.bottom(down3)
# Expanding Path
up1 = model.up1(torch.cat([down3, bottom], 1))
up2 = model.up2(torch.cat([down2, up1], 1))
up3 = model.up3(torch.cat([down1, up2], 1))
# Final output
final = model.final(up3)
print(final.shape)

print("#########################################################")
