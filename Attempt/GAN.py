import torch
import torch.nn as nn
import torch.optim as optim
# Define the generator network
class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        return x

# Define the discriminator network
class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

# Initialize the generator and discriminator
input_dim = 100
output_dim = 784
generator = Generator(input_dim, output_dim)
discriminator = Discriminator(output_dim)

# Define the loss function and optimizer
criterion = nn.BCELoss()
generator_optimizer = optim.Adam(generator.parameters(), lr=0.001)
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    for batch in range(num_batches):
        # Train the discriminator
        real_data = get_real_data(batch_size)  # Replace with your own data loading function
        fake_data = generator(torch.randn(batch_size, input_dim))
        discriminator_real_output = discriminator(real_data)
        discriminator_fake_output = discriminator(fake_data.detach())
        discriminator_real_loss = criterion(discriminator_real_output, torch.ones_like(discriminator_real_output))
        discriminator_fake_loss = criterion(discriminator_fake_output, torch.zeros_like(discriminator_fake_output))
        discriminator_loss = discriminator_real_loss + discriminator_fake_loss
        discriminator_optimizer.zero_grad()
        discriminator_loss.backward()
        discriminator_optimizer.step()

        # Train the generator
        fake_data = generator(torch.randn(batch_size, input_dim))
        discriminator_fake_output = discriminator(fake_data)
        generator_loss = criterion(discriminator_fake_output, torch.ones_like(discriminator_fake_output))
        generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_optimizer.step()

    # Print the loss for every epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Generator Loss: {generator_loss.item()}, Discriminator Loss: {discriminator_loss.item()}")
