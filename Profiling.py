import torch
import torch.nn as nn
import torch.optim as optim
from torch.profiler import profile, record_function, ProfilerActivity

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(100, 10)

    def forward(self, x):
        return self.fc(x)

# Create model and move to MPS device if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)
model = SimpleModel().to(device)

# Increase batch size here
batch_size = 1024  # Changed from 32 to 128

# Create dummy data with increased batch size
input_data = torch.randn(batch_size, 100, device=device)
target = torch.randint(0, 10, (batch_size,), device=device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function
def train_batch():
    model.train()
    optimizer.zero_grad(set_to_none=True)
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

# Profiling
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True,
             profile_memory=True,
             with_stack=True,
             use_cuda=torch.backends.mps.is_available()) as prof:
    for _ in range(1000):  # 100 iterations for better averaging
        with record_function("train_batch"):
            train_batch()

# Print profiling results
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))

# Save profiling results to a file
prof.export_chrome_trace("pytorch_trace.json")
