# Frozen CLIP ViT-B/32 spatial-feature contract

Date: 2026-10-03

Status: preregistered before Stage R feature extraction or training

This contract exposes spatial tokens from the exact frozen visual encoder that
produced the existing D13 global features. It does not introduce a new encoder,
new preprocessing, privileged localization, or CLIP fine-tuning.

## Checkpoint identity

| Item | Frozen value |
|---|---|
| Model | `openai/clip-vit-base-patch32` |
| Revision | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` |
| PyTorch checkpoint SHA-256 | `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` |
| Config SHA-256 | `b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770` |
| Preprocessor SHA-256 | `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd` |
| D13 store SHA-256 | `e5599153cf44282242d955be62249bbafc752b782f5ae76a0ec0d58f9146f98e` |
| D13 manifest SHA-256 | `7c374d187e06c6ef99851c159134e722fa96acca9d69461bd1541e1a2895b0dc` |

The local checkpoint is loaded with `local_files_only=True`. Every CLIP
parameter has `requires_grad=False`; extraction runs under inference mode and
no optimizer ever receives an encoder parameter.

## Exact image path

The source is the upright `128x128x3` uint8 RGB array at the exact HDF5 path
`data/demo_{id}/obs/agentview_rgb`. There is no vertical flip, modified RGB,
masking, object crop, or augmentation. The pinned `CLIPProcessor` converts RGB
to the model input by bicubic resize to 224, 224 center crop, conversion to
float, and the checkpoint's frozen CLIP channel mean and standard deviation.

The ViT image contract is:

```text
pixel_values                         [B,3,224,224]
Conv2d(kernel=32,stride=32)          [B,768,7,7]
row-major flatten                    [B,49,768]
prepend learned CLS                  [B,50,768]
add learned positional embeddings   [B,50,768]
pre-layer normalization              [B,50,768]
12-layer frozen transformer          [B,50,768]
```

The checkpoint has patch size 32, hidden width 768, 12 attention heads, 12
transformer layers, intermediate width 3072, 50 learned visual positions, and
projection width 512. The 49 spatial tokens are not inferred from a name or
assumed from prior reports; they follow from the pinned config and the observed
encoder output shape.

## Global path and spatial path

The existing global D13 vector follows the checkpoint's standard path:

```text
encoded CLS token
  -> vision post-layernorm
  -> shared visual_projection 768x512
  -> L2 normalization
  -> global feature [512]
```

The Stage R patch path deliberately reuses the same post-encoder normalization
and projection parameters:

```text
encoded spatial tokens positions 1..49
  -> the same vision post-layernorm
  -> the same visual_projection 768x512
  -> per-token L2 normalization
  -> patch feature grid [49,512]
```

CLS is excluded from the patch grid. The learned positional embeddings have
already entered before the transformer and are retained implicitly in each
spatial token. No separate global vector is concatenated into R1--R3 unless a
family definition explicitly says so; the registered R1--R3 definitions do
not.

## Fixed two-dimensional coordinates

The token order is the ViT row-major 7x7 order. For row `i` and column `j`, the
only added coordinate is the fixed image-grid cell center:

```text
x_j = 2 * (j + 0.5) / 7 - 1
y_i = 2 * (i + 0.5) / 7 - 1
```

These coordinates range from `-6/7` to `+6/7`. They describe only location in
the processed image grid. They are not camera rays, simulator coordinates,
object positions, plate positions, or projections of privileged state.

## Deterministic extraction contract

Calibration is traversed as task0--9, then demo31--34, then native timestep.
Images are encoded in consecutive batches of 64, matching the original D13
extraction convention. CUDA uses float16 autocast and the persisted patch grid
uses float16; downstream heads consume float32 copies. A fixed repeated sample
must produce bitwise-identical persisted float16 tokens in the same process.

The extractor must additionally establish all of the following before Stage R
training:

- every patch and reconstructed global feature is finite;
- every patch tensor has exact shape `[frames,49,512]`;
- a repeated fixed batch is exactly equal after float16 persistence;
- the reconstructed global vector has minimum cosine at least `0.9999` and
  maximum absolute difference at most `0.002` versus the frozen D13 vector;
- zero CLIP parameters are trainable and no CLIP gradient exists;
- the access ledger contains only demo31--34 source RGB plus the frozen D13
  calibration `vision` arrays and one global language matrix;
- demo36--39 and demo40--49 have zero numeric and metadata access in Stage R.

Failure of any integrity condition stops Stage R as a data/implementation
failure; it cannot be converted into evidence for or against spatial
representation sufficiency.

## Deployment-input boundary

Allowed deployable inputs are the natural RGB-derived frozen patch grid, its
fixed image coordinates, and the frozen task-language embedding. Frozen T1,
G2, object identity, masks, boxes, plate location, simulator state, object
pose, camera projection, proprioception, time, action, reward, done, success,
and task ID may be used only where already authorized for labels, strata, or
integrity checks; none may enter R0--R3.

Attention entropy and centers are diagnostics. They are not object masks,
localization claims, or proof that either R3 slot corresponds to a bowl or a
plate.
