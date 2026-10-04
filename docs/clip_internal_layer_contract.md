# Frozen CLIP ViT-B/32 internal spatial-layer contract

Date: 2026-10-04

This contract records the actual installed tensor path used by Stage I. It is
not inferred from a generic ViT diagram. The environment uses
`transformers==4.21.1`; the inspected source file has SHA-256
`d0c8e6ce940205273debcdb7737c8654dae183ef4a328bc76455dc2b8e9b3762`.

## Actual implementation path

The installed `CLIPVisionEmbeddings.forward` applies the frozen 32-by-32 patch
convolution, flattens the 7-by-7 grid in row-major order, prepends the class
token, and adds the learned positional embedding. `CLIPVisionTransformer`
then applies `pre_layrnorm` before invoking its twelve encoder blocks.

When `output_hidden_states=True`, the installed encoder records the current
tensor immediately before each block and appends the final tensor after block
12. Consequently:

- `hidden_states[0]` is after `pre_layrnorm`, before block 1;
- `hidden_states[3]` is after block 3;
- `hidden_states[6]` is after block 6;
- `hidden_states[9]` is after block 9;
- `hidden_states[12]` is after block 12.

The stock forward path applies `post_layernorm` only to the pooled class token.
The existing project patch contract deliberately applies that same frozen
layer norm to every final patch token, then the shared 768-to-512 visual
projection and per-token L2 normalization. Stage I preserves that exact
baseline as `L12-proj`.

## Frozen extraction points

| Name | Exact tensor | Patch dimension |
|---|---|---:|
| L0 | `hidden_states[0][:,1:,:]` | 768 |
| L3 | `hidden_states[3][:,1:,:]` | 768 |
| L6 | `hidden_states[6][:,1:,:]` | 768 |
| L9 | `hidden_states[9][:,1:,:]` | 768 |
| L12-pre | `hidden_states[12][:,1:,:]` | 768 |
| L12-postLN | `post_layernorm(hidden_states[12][:,1:,:])` | 768 |
| L12-proj | `normalize(visual_projection(L12-postLN))` | 512 |

The raw embedding tensor before `pre_layrnorm` is inspected for implementation
integrity but is not an eighth candidate. `L0` is the actual block-1 input and
the installed encoder's canonical zero-depth hidden state. This avoids an
unregistered layer-zoo expansion.

Every candidate excludes CLS, retains exactly 49 tokens, and uses the same
fixed row-major cell-center coordinates in `[-1,1]^2`. No object mask, box,
G2 identity, simulator coordinate, plate coordinate, or object pose enters an
extraction or reader.

## Fair reader boundary

Every layer is standardized using demo31--33 only and is evaluated with the
same two bounded readers: P0 task-independent single-slot pooling and P1
language-conditioned single-slot pooling. Both use 24-dimensional keys and
values, the same 64-unit antisymmetric pair head, identical training pairs,
optimizer, epochs, temperature grid, and parameter ceiling. Only the frozen
token dimension differs (768 versus the exact 512-dimensional projected
baseline).

The formal Stage-I material comparison is P1 internal layer versus P1
`L12-proj`. Historical R0 global and prior R3 values remain immutable external
references. This isolates extraction depth from architecture changes.
