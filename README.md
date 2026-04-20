Self-Pruning Neural Network on CIFAR-10
Tredence Analytics — AI Engineering Intern Case Study

Overview
This project implements a feed-forward neural network that prunes itself during training on the CIFAR-10 image classification dataset.
Instead of a post-training pruning step, the network uses learnable gate parameters associated with every weight. An L1 sparsity regularization loss encourages these gates to become exactly zero — effectively removing unnecessary weights on the fly.

Project Structure
tredence-self-pruning-network/
├── self_pruning_network.py         # Main Python script (all parts)
├── report.md                       # Case study report with results & analysis
├── training_curves.png             # Loss curves for all 3 λ values
└── gate_distribution_lambda_5e-05.png  # Gate value distribution plot

How It Works
1. PrunableLinear Layer
A custom replacement for nn.Linear with an extra learnable gate_scores parameter:
pythongates = sigmoid(gate_scores)          # values in (0, 1)
pruned_weights = weight * gates       # element-wise masking
output = input @ pruned_weights.T + bias
When a gate → 0, its corresponding weight is silenced (pruned).
2. Sparsity Loss (L1)
Total Loss = CrossEntropyLoss + λ × Σ(all gate values)
The L1 penalty has a constant gradient near zero, which drives gates to exactly zero — unlike L2 which only shrinks values without zeroing them.
3. λ Controls the Tradeoff
λ ValueEffectLow (1e-6)Minimal pruning, standard network behaviourMedium (5e-5)Balanced pruning with no accuracy lossHigh (1e-4)Aggressive pruning, acts as regularization

Results
Lambda (λ)Test AccuracySparsity Level (%)1e-6 (low)54.29%0.03%5e-5 (medium)54.91%25.54%1e-4 (high)55.37%42.39%

At λ = 1e-4, 42% of weights were pruned while accuracy actually increased — confirming that many weights are redundant and pruning acts as effective regularization.


Setup & Run
Prerequisites

Python 3.10 or 3.11
pip

Install dependencies
bashpip install torch torchvision matplotlib numpy
Run the script
bashpython self_pruning_network.py
CIFAR-10 (~170MB) downloads automatically on first run.
Expected runtime: ~1.5–2 hours on CPU
Output

Prints epoch-by-epoch loss for each λ value
Prints final results summary table
Saves training_curves.png and gate_distribution_lambda_*.png


Key Concepts
Why L1 encourages sparsity:
L1 applies a constant gradient (-λ) to every gate regardless of its size. Even very small gate values keep getting pushed toward zero until they reach exactly 0. L2, by contrast, has a gradient proportional to the value — so near zero it effectively stops pushing, leaving weights small but never truly zero.
Gradient flow:
Both weight and gate_scores are registered nn.Parameter objects. PyTorch's autograd handles backpropagation through the sigmoid and element-wise multiplication automatically, updating both sets of parameters every step.

Tech Stack

PyTorch — custom layer, training loop, autograd
Torchvision — CIFAR-10 dataset
Matplotlib — result plots
NumPy — gate analysis
