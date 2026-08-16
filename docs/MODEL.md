# Model

PFIR-SAM2 is a prompt-free dense instance-segmentation model for bright-field
organoid images.

## Architecture

- The global branch loads the official SAM2.1 Hiera-L image encoder.
- The prompt encoder, native SAM2 mask decoder, and memory path are not used.
- LoRA with rank 8, alpha 16, and dropout 0 wraps 195 matched `Linear` modules
  in the retained model. All original SAM2 parameters are frozen; LoRA
  parameters remain trainable.
- A parallel CNN branch returns local features at 1/4, 1/8, and 1/16 scale.
- SAM2 and CNN features are projected and added at those three scales only.
- Residual coarse-to-fine fusion produces F16, F8, and F4 features.
- A channel-specific sigmoid gate modulates F4 as `F4 * (1 + gate)`.
- Two subsequent upsampling/refinement stages return full-resolution features.
- Three independent 1x1 heads output foreground, boundary, and center logits.

The retained implementation is exposed through `pfir_sam2/model.py`; the exact
training implementation is kept in `pfir_sam2/core_training.py` so checkpoint
parameter names and mathematical behavior remain unchanged.

## Cue-guided instance reconstruction

Sliding-window logits are sigmoid-transformed and Hann-blended. The frozen
final-v2 reconstruction uses:

```text
foreground threshold = 0.45
boundary threshold = 0.40
center threshold = 0.25
center Gaussian sigma = 1
minimum peak distance = 10 pixels
D_norm = EDT(foreground) / (max(EDT(foreground)) + 1e-6)
S = 0.7 * smoothed_center + 0.3 * D_norm
watershed elevation = -S
```

Boundary probability does not enter the main score. It defines an interior
candidate only when no center-derived marker exists anywhere in the image. If
that fallback also produces no marker, foreground connected components are
returned.

Small-object rescue starts from the reconstruction, checks raw candidate areas
from 5 to 500 pixels, requires IoU below 0.05 and containment below 0.30, adds
at least 5 free pixels, and finally relabels instances sequentially. In the
retained inference chain, raw candidates themselves are generated with a
30-pixel minimum-area filter.
