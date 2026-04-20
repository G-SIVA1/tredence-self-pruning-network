"""
Self-Pruning Neural Network on CIFAR-10
========================================
Tredence Analytics - AI Engineering Intern Case Study

This script implements a feed-forward neural network that prunes itself
during training using learnable gate parameters and L1 sparsity regularization.

Author: [Your Name]
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

# ─────────────────────────────────────────────────────────────────────────────
# DEVICE SETUP
# ─────────────────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: PrunableLinear Layer
# ─────────────────────────────────────────────────────────────────────────────

class PrunableLinear(nn.Module):
    """
    A custom Linear layer augmented with learnable gate_scores.

    Each weight has a corresponding gate (sigmoid(gate_score)) that scales it.
    When a gate approaches 0, the weight is effectively pruned.

    Forward pass:
        gates        = sigmoid(gate_scores)          # values in (0, 1)
        pruned_weights = weight * gates              # element-wise
        output         = input @ pruned_weights.T + bias
    """

    def __init__(self, in_features: int, out_features: int):
        super(PrunableLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Standard weight and bias parameters (same as nn.Linear)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias   = nn.Parameter(torch.zeros(out_features))

        # Learnable gate scores — same shape as weight
        # These will be pushed toward -∞ by L1 loss → sigmoid → 0 (pruned)
        self.gate_scores = nn.Parameter(torch.zeros(out_features, in_features))

        # Initialize weights using Kaiming uniform (standard practice)
        nn.init.kaiming_uniform_(self.weight, a=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Convert gate_scores → gates in range (0, 1)
        gates = torch.sigmoid(self.gate_scores)

        # Step 2: Element-wise multiply weights by gates
        #         If gate → 0, the weight is "pruned" (silenced)
        pruned_weights = self.weight * gates

        # Step 3: Standard linear operation: y = x @ W^T + b
        #         Using F.linear equivalent written manually for clarity
        return x @ pruned_weights.t() + self.bias

    def get_gates(self) -> torch.Tensor:
        """Return the current gate values (detached from graph, for analysis)."""
        return torch.sigmoid(self.gate_scores).detach()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of all gate values in this layer (used in total loss)."""
        return torch.sigmoid(self.gate_scores).abs().sum()


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL NETWORK DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

class SelfPruningNet(nn.Module):
    """
    A feed-forward network for CIFAR-10 image classification.

    Architecture:
        Input (3×32×32 = 3072)
        → PrunableLinear(3072, 512) → ReLU
        → PrunableLinear(512, 256)  → ReLU
        → PrunableLinear(256, 128)  → ReLU
        → PrunableLinear(128, 10)   → (logits for 10 classes)

    All linear layers are PrunableLinear so gates are learned throughout.
    """

    def __init__(self):
        super(SelfPruningNet, self).__init__()

        self.layers = nn.Sequential(
            PrunableLinear(3072, 512),
            nn.ReLU(),
            PrunableLinear(512, 256),
            nn.ReLU(),
            PrunableLinear(256, 128),
            nn.ReLU(),
            PrunableLinear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Flatten image: (batch, 3, 32, 32) → (batch, 3072)
        x = x.view(x.size(0), -1)
        return self.layers(x)

    def total_sparsity_loss(self) -> torch.Tensor:
        """
        Sum L1 loss over all PrunableLinear layers.
        This is the SparsityLoss term in: Total Loss = CE + λ * SparsityLoss
        """
        loss = torch.tensor(0.0, device=device)
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                loss = loss + module.sparsity_loss()
        return loss

    def get_all_gates(self) -> np.ndarray:
        """Collect all gate values from every PrunableLinear layer."""
        all_gates = []
        for module in self.modules():
            if isinstance(module, PrunableLinear):
                all_gates.append(module.get_gates().cpu().numpy().flatten())
        return np.concatenate(all_gates)

    def compute_sparsity_level(self, threshold: float = 1e-2) -> float:
        """
        Compute % of weights whose gate value is below the threshold.
        These are considered 'pruned'.
        """
        all_gates = self.get_all_gates()
        pruned = np.sum(all_gates < threshold)
        return 100.0 * pruned / len(all_gates)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def get_cifar10_loaders(batch_size: int = 128):
    """Download and return CIFAR-10 train and test DataLoaders."""

    # Normalize to mean=0.5, std=0.5 per channel (standard for CIFAR-10)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  num_workers=2)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# PART 3: TRAINING LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_model(lambda_sparse: float,
                num_epochs: int = 15,
                lr: float = 1e-3,
                batch_size: int = 128):
    """
    Train the SelfPruningNet with a given sparsity regularization strength λ.

    Total Loss = CrossEntropyLoss + λ × SparsityLoss (L1 of all gates)

    Args:
        lambda_sparse : Weight of the sparsity penalty (λ)
        num_epochs    : Number of training epochs
        lr            : Learning rate for Adam optimizer
        batch_size    : Mini-batch size

    Returns:
        model         : Trained model
        train_losses  : List of total loss per epoch
    """

    train_loader, test_loader = get_cifar10_loaders(batch_size)

    model = SelfPruningNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []

    print(f"\n{'='*60}")
    print(f"Training with λ = {lambda_sparse}")
    print(f"{'='*60}")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Classification loss (Cross-Entropy)
            ce_loss = criterion(outputs, labels)

            # Sparsity loss (L1 on all gate values)
            sp_loss = model.total_sparsity_loss()

            # Total loss — this is what we optimize
            total_loss = ce_loss + lambda_sparse * sp_loss

            # Backward pass — gradients flow through both weights AND gate_scores
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"  Epoch [{epoch+1:02d}/{num_epochs}]  Loss: {avg_loss:.4f}")

    # Evaluate on test set
    test_accuracy = evaluate_model(model, test_loader)
    sparsity = model.compute_sparsity_level(threshold=1e-2)

    print(f"\n  ✓ Test Accuracy : {test_accuracy:.2f}%")
    print(f"  ✓ Sparsity Level: {sparsity:.2f}%  (gates < 0.01 threshold)")

    return model, train_losses, test_accuracy, sparsity


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(model: nn.Module, test_loader: DataLoader) -> float:
    """Evaluate model accuracy on the test set."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return 100.0 * correct / total


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_gate_distribution(model: nn.Module, lambda_val: float):
    """
    Plot the distribution of final gate values for the trained model.

    A successful pruning result shows:
      - A large spike near 0   (pruned weights)
      - A cluster near 0.5–1  (active weights)
    """
    all_gates = model.get_all_gates()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(all_gates, bins=100, color="#2563EB", edgecolor="white", alpha=0.85)
    ax.set_title(f"Gate Value Distribution — λ = {lambda_val}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Gate Value (sigmoid output)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.axvline(x=0.01, color="red", linestyle="--", linewidth=1.5, label="Pruning threshold (0.01)")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    filename = f"gate_distribution_lambda_{str(lambda_val).replace('.', '_')}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"  Plot saved → {filename}")


def plot_training_curves(all_losses: dict):
    """Plot training loss curves for all λ values on one chart."""
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#2563EB", "#16A34A", "#DC2626"]

    for (lam, losses), color in zip(all_losses.items(), colors):
        ax.plot(range(1, len(losses) + 1), losses, label=f"λ = {lam}", color=color, linewidth=2)

    ax.set_title("Training Loss per Epoch (all λ values)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Total Loss", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    plt.show()
    print("  Plot saved → training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN — Run experiments for 3 values of λ
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Three λ values: low, medium, high
    # Low  → less pruning, better accuracy
    # High → more pruning, lower accuracy
    lambda_values = [1e-6, 5e-5, 1e-4]

    results = {}       # { lambda: (accuracy, sparsity) }
    all_losses = {}    # { lambda: [loss per epoch] }
    best_model = None
    best_lambda = None

    for lam in lambda_values:
        model, losses, accuracy, sparsity = train_model(
            lambda_sparse=lam,
            num_epochs=25,
            lr=1e-3,
            batch_size=128,
        )
        results[lam] = (accuracy, sparsity)
        all_losses[lam] = losses

        # Track the medium λ as "best" for the gate distribution plot
        if lam == lambda_values[1]:
            best_model = model
            best_lambda = lam

    # ── Print Summary Table ───────────────────────────────────────────────────
    print("\n\n" + "="*55)
    print(f"{'RESULTS SUMMARY':^55}")
    print("="*55)
    print(f"{'Lambda':<15} {'Test Accuracy':>15} {'Sparsity Level':>15}")
    print("-"*55)
    for lam, (acc, spar) in results.items():
        print(f"{lam:<15} {acc:>14.2f}% {spar:>14.2f}%")
    print("="*55)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_training_curves(all_losses)
    plot_gate_distribution(best_model, best_lambda)

    print("\nDone! Check the saved .png files for plots.")
