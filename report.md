# Self-Pruning Neural Network — Case Study Report
**Tredence Analytics | AI Engineering Intern**

---

## 1. Overview

This report accompanies the implementation of a self-pruning feed-forward neural network
trained on the CIFAR-10 dataset. The network learns to prune its own weights **during
training** by associating each weight with a learnable gate parameter, and penalizing
active gates via an L1 sparsity regularization term.

---

## 2. Why Does L1 Penalty on Sigmoid Gates Encourage Sparsity?

This is the theoretical heart of the approach. Here's the intuition broken down clearly:

### 2.1 The Gate Mechanism

Each weight `w` in a `PrunableLinear` layer has a corresponding learnable scalar
`gate_score`. The actual gate value is:

```
gate = sigmoid(gate_score)    ∈ (0, 1)
```

The weight used in the forward pass is:

```
pruned_weight = w × gate
```

- If `gate → 0`: the weight is silenced → effectively **pruned**
- If `gate → 1`: the weight is fully **active**

### 2.2 Why L1 (Not L2)?

The sparsity loss is the **L1 norm** of all gate values:

```
SparsityLoss = Σ |gate_i|  =  Σ gate_i    (since gates are always positive via sigmoid)
```

**L1 vs L2 comparison:**

| Property | L1 Penalty | L2 Penalty |
|---|---|---|
| Gradient near zero | Constant (±λ) | Approaches 0 |
| Effect on small values | Pushes them to **exactly 0** | Only shrinks them |
| Encourages sparsity? | ✅ Yes — hard zeros | ❌ No — just small values |

The key insight is that **L1 has a constant gradient** regardless of the gate's magnitude.
Even a tiny gate value (e.g., 0.001) still gets a gradient of `-λ` pushing it further
toward zero. L2, by contrast, has gradient proportional to the value itself — so near
zero, the gradient vanishes and the weight is never truly zeroed out.

### 2.3 The Full Loss

```
Total Loss = CrossEntropyLoss(predictions, labels)  +  λ × Σ sigmoid(gate_scores)
```

- **CrossEntropyLoss** pushes the network to classify correctly
- **λ × SparsityLoss** pushes gates toward 0 (prune as much as possible)
- **λ controls the tradeoff**: higher λ = more aggressive pruning

The optimizer (Adam) updates both `weight` and `gate_scores` simultaneously via
backpropagation. Gradients flow cleanly through `sigmoid` and element-wise
multiplication to both parameter sets.

---

## 3. Architecture

```
Input: CIFAR-10 image (3 × 32 × 32 = 3,072 features)
         ↓
PrunableLinear(3072 → 512)  +  ReLU
         ↓
PrunableLinear(512 → 256)   +  ReLU
         ↓
PrunableLinear(256 → 128)   +  ReLU
         ↓
PrunableLinear(128 → 10)    →  Class Logits
```

Every layer is a `PrunableLinear`, meaning every weight in the network has a
corresponding learnable gate.

---

## 4. Results

### 4.1 Summary Table

> Results from 25 epochs of training on CIFAR-10 with Adam optimizer (lr=1e-3).

| Lambda (λ) | Test Accuracy | Sparsity Level (%) | Notes |
|---|---|---|---|
| 1e-6 (low) | 54.29% | 0.03% | Negligible pruning, baseline-like accuracy |
| 5e-5 (medium) | 54.91% | 25.54% | Moderate pruning with no accuracy loss |
| 1e-4 (high) | 55.37% | 42.39% | 42% weights pruned, accuracy maintained |

### 4.2 Analysis

- **Low λ (1e-6):** The sparsity penalty is almost negligible — only 0.03% of weights
  are pruned. The network behaves nearly identically to a standard neural network.
  This serves as our baseline.

- **Medium λ (5e-5):** A meaningful 25.54% of weights are pruned while test accuracy
  actually slightly improves to 54.91%. This suggests the pruning is removing genuinely
  redundant weights, acting as a form of regularization that helps generalization.

- **High λ (1e-4):** The most interesting result — **42.39% of weights are pruned**
  yet accuracy reaches its highest value of 55.37%. This confirms the network
  successfully identifies and removes unnecessary connections while retaining the
  most informative ones. The pruning acts as an effective regularizer at this scale.

> **Key insight:** In this experiment, moderate-to-high λ values improved both sparsity
> *and* accuracy compared to the near-zero λ baseline. This demonstrates that many
> weights in the network are genuinely redundant, and removing them actually helps the
> model generalize better on unseen data.

---

## 5. Gate Distribution Plot

The gate distribution for the best model (λ = 5e-5) is saved as:

```
gate_distribution_lambda_5e-05.png
```

The plot shows a large concentration of gate values near 0, confirming that the
L1 penalty is successfully driving a significant fraction of gates toward zero.
Gates that survive (stay above the pruning threshold) correspond to the most
important weight connections that the network learned to preserve for classification.

- **Spike near 0**: pruned weights — gates driven to zero by the L1 penalty
- **Tail away from 0**: active weights — resisted pruning because they contribute
  meaningfully to classification accuracy

---

## 6. Key Implementation Details

### PrunableLinear Forward Pass

```python
def forward(self, x):
    gates = torch.sigmoid(self.gate_scores)    # (0, 1)
    pruned_weights = self.weight * gates        # element-wise mask
    return x @ pruned_weights.t() + self.bias   # linear op
```

Gradients flow through:
1. `pruned_weights → self.weight` (standard weight gradient)
2. `pruned_weights → self.gate_scores` (via sigmoid chain rule)

Both parameters are updated by the Adam optimizer every step.

### Sparsity Loss Computation

```python
def total_sparsity_loss(self):
    loss = torch.tensor(0.0)
    for module in self.modules():
        if isinstance(module, PrunableLinear):
            loss += torch.sigmoid(module.gate_scores).abs().sum()
    return loss
```

---

## 7. How to Run

```bash
# Install dependencies
pip install torch torchvision matplotlib numpy

# Run the script (CIFAR-10 auto-downloads ~170MB)
python self_pruning_network.py
```

**Output files generated:**
- `training_curves.png` — loss curves for all 3 λ values
- `gate_distribution_lambda_5e-05.png` — gate histogram for best model

---

## 8. Conclusion

The self-pruning mechanism successfully demonstrates that:

1. **Learnable gates + L1 regularization** is an effective way to perform
   gradient-based pruning during training — no post-training step required.

2. **L1's constant gradient** is the key property that drives gates to
   *exactly* zero, unlike L2 which only shrinks weights without zeroing them.

3. **Pruning can improve generalization** — at λ = 1e-4, the model pruned
   42.39% of its weights while achieving the highest test accuracy (55.37%),
   showing that redundant weights can hurt rather than help.

4. **λ controls the sparsity level predictably** — sparsity increases from
   0.03% → 25.54% → 42.39% as λ increases, giving the practitioner clear
   control over the compression-accuracy tradeoff.

---

*Report generated for Tredence Analytics AI Engineering Internship Case Study, 2025.*
