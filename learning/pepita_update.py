"""
Post-Backprop 4B — PEPITA Weight Updates
==========================================
PEPITA (Parameter-free Error Propagation for Intelligent Target
Assignment) local difference learning.

For each layer l, training proceeds in two forward passes per batch:

  1. FREE PASS:
       h_l = Layer_l(h_{l-1})
       Compute local error: e_l = softmax(Head_l(h_l)) - one_hot(y)

  2. PERTURBED PASS:
       h'_{l-1} = h_{l-1} + scale · F_l · e_l
       h'_l = Layer_l(h'_{l-1})
     where F_l is a fixed random projection (d_model × vocab_size).

  3. WEIGHT UPDATE:
       For each weight W in layer l:
         ΔW = (η / B) · input_free^T @ (output_perturbed - output_free)

This guarantees non-zero updates using strictly local signals and
zero autograd.

References:
  - Dellaferrera & Bhalerao (2022): "Error-driven Input Modulation:
    Solving the Credit Assignment Problem without a Backward Pass"
  - Xie & Seung (2003): "Equivalence of Backpropagation and
    Contrastive Hebbian Learning in a Layered Network"
"""

import torch


class PEPITAUpdater:
    """Manages PEPITA-style local difference learning for one layer.

    Holds:
      - F_l: a fixed random projection matrix (d_model × vocab_size)
             used to project the error signal into the input space.
      - Momentum buffers (optional) for each weight matrix.
    """

    def __init__(self, d_model: int, vocab_size: int,
                 perturbation_scale: float = 0.1,
                 dtype: torch.dtype = torch.float16,
                 seed: int = 0):
        """
        Args:
            d_model:            hidden dimension.
            vocab_size:         vocabulary size (error signal dimension).
            perturbation_scale: scalar multiplier for the perturbation.
            dtype:              tensor dtype.
            seed:               random seed for F_l reproducibility.
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.scale = perturbation_scale

        # Fixed random projection: (d_model, vocab_size)
        # Initialized from N(0, 1/sqrt(vocab_size)) for scale stability
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.F_proj = torch.randn(
            d_model, vocab_size, dtype=dtype, generator=gen
        ) / (vocab_size ** 0.5)

        # Momentum buffers — lazily initialised
        self._momentum = {}
        self._momentum_beta = 0.9

    def perturb_input(self, h_prev: torch.Tensor,
                      error: torch.Tensor) -> torch.Tensor:
        """Create the perturbed input for the second forward pass.

        h'_{l-1} = h_{l-1} + scale · error @ F_l^T

        Args:
            h_prev: (B, T, D)  original input to this layer.
            error:  (B, T, V)  local error from the prediction head.
        Returns:
            h_perturbed: (B, T, D) perturbed input.
        """
        # error: (B, T, V) @ F_proj^T: (V, D) → (B, T, D)
        # Compute in fp32 to prevent overflow (V=50257 is a large reduction dim)
        perturbation = (error.float() @ self.F_proj.float().T).to(h_prev.dtype)  # (B, T, D)
        return h_prev + self.scale * perturbation

    def compute_weight_updates(
        self,
        free_cache: dict,
        perturbed_cache: dict,
        weight_pairs: list,
        lr: float,
    ) -> dict:
        """Compute and apply PEPITA updates for all weights in the layer.

        For each weight matrix W with pre-synaptic key `pre` and
        post-synaptic key `post`:

          ΔW = (η / N) · free_cache[pre]^T @ (perturbed_cache[post] - free_cache[post])

        Then normalise by RMS for scale-invariance (LN-CHL style).

        Args:
            free_cache:      dict of intermediates from the free pass.
            perturbed_cache: dict of intermediates from the perturbed pass.
            weight_pairs:    list of (name, W_tensor, pre_key, post_key).
            lr:              learning rate.
        Returns:
            update_norms: dict mapping weight name → L2 norm of the update.
        """
        update_norms = {}

        for name, W, pre_key, post_key in weight_pairs:
            # Pre-synaptic activations from the FREE pass
            pre = free_cache[pre_key].float()           # (B, T, d_in)
            # Post-synaptic difference
            post_free = free_cache[post_key].float()    # (B, T, d_out)
            post_pert = perturbed_cache[post_key].float()
            delta_post = post_pert - post_free          # (B, T, d_out)

            # Flatten batch and sequence dimensions
            N = pre.shape[0] * pre.shape[1]
            pre_flat = pre.reshape(N, -1)               # (N, d_in)
            delta_flat = delta_post.reshape(N, -1)      # (N, d_out)

            # Raw update: cross-correlation of input with output difference
            raw_update = pre_flat.T @ delta_flat / N     # (d_in, d_out)

            # Clamp update elements to prevent anomalous steps
            raw_update.clamp_(min=-0.1, max=0.1)

            # Optional momentum
            if name not in self._momentum:
                self._momentum[name] = torch.zeros_like(raw_update)
            self._momentum[name] = (
                self._momentum_beta * self._momentum[name]
                + (1 - self._momentum_beta) * raw_update
            )

            # Weight decay (L2 regularisation) to keep parameters bounded
            W.data.mul_(1.0 - lr * 1e-4)

            # Apply update (in-place, no autograd)
            final_update = lr * self._momentum[name].to(W.dtype)
            W.data.add_(final_update)

            update_norms[name] = final_update.norm().item()

        return update_norms

    def to(self, device):
        self.F_proj = self.F_proj.to(device)
        for k in self._momentum:
            self._momentum[k] = self._momentum[k].to(device)
        return self


def compute_norm_updates(norm_layer, h_free: torch.Tensor,
                         h_perturbed: torch.Tensor,
                         lr: float) -> float:
    """Update RMSNorm gain using a simple local Hebbian-like rule.

    The gain is adjusted based on the average absolute difference
    between free and perturbed outputs at each dimension:

      Δgain_d = lr · mean_over_batch(|h_perturbed_d| - |h_free_d|)

    This nudges the gain toward dimensions where the perturbation
    had a larger effect, which is where the error signal is strongest.

    Args:
        norm_layer: RMSNorm instance.
        h_free:     (B, T, D) free-pass activations after this norm.
        h_perturbed:(B, T, D) perturbed-pass activations after this norm.
        lr:         learning rate.
    Returns:
        update_norm: scalar L2 norm of the gain update.
    """
    diff = h_perturbed.float().abs().mean(dim=(0, 1)) - \
           h_free.float().abs().mean(dim=(0, 1))      # (D,)
    
    # Clamp update elements to prevent anomalous steps
    diff.clamp_(min=-0.1, max=0.1)
    
    update = lr * diff
    norm_layer.gain.data.add_(update.to(norm_layer.gain.dtype))
    return update.norm().item()
