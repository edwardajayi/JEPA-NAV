# What the V-JEPA 2 Phase A Command Does — Mathematical View

Command:

```bash
cd /ocean/projects/cis250225p/eajayi1/JEPANAV
python scripts/run_vjepa2_on_frames.py --frames_dir data/Replica-Dataset/assets/ --num_frames 16
```

Below is what this pipeline does in order, then a compact mathematical description.

---

## 1. What the code does (step-by-step)

1. **Load images**  
   Reads all PNG/JPG in `data/Replica-Dataset/assets/` in filename order. You have 4 images; the script uses them and **pads to 16 frames** by repeating the last frame so the clip has exactly `T = 16` frames.

2. **Build a video tensor**  
   Stacks frames into a single array:
   - Shape: **`(T, C, H, W)`** = (16, 3, H, W) with H, W the image height/width (e.g. 480×640).  
   - So the “video” is 16 time steps, 3 RGB channels, and two spatial dimensions.

3. **Preprocess (processor)**  
   The V-JEPA 2 **processor**:
   - Resizes/crops each frame to the model’s **crop size** (e.g. 256×256).  
   - Normalizes pixel values (e.g. to zero mean, unit variance or to a fixed range).  
   - Produces a tensor **`x`** that the model expects: typically **`(B, T, C, H_c, W_c)`** with B=1, T=16, C=3, H_c=W_c=256.

4. **Encoder forward pass**  
   **V-JEPA 2 encoder** (Vision Transformer over space-time):
   - Treats the clip as a **sequence of spatiotemporal “patches” (tubelets)**.  
   - Runs a **Transformer encoder** on this sequence.  
   - Outputs a sequence of **hidden vectors** (one per patch/token): **`last_hidden_state`**.

5. **What you get**  
   - **`last_hidden_state`**: shape **`(1, N, D)`**  
     - 1 = batch size  
     - N = number of tokens (patch/tubelets)  
     - D = hidden size (e.g. 1024 for ViT-L).  
   - This is the **embedding** of your 16-frame clip: a set of D-dimensional vectors you can pool (e.g. mean over N) to get one vector per clip for downstream use (e.g. policy input).

So in one sentence: **the command turns 4 (padded to 16) Replica asset images into a 256×256 video clip, runs the V-JEPA 2 encoder on it, and produces a sequence of 1024-d embedding vectors.**

---

## 2. Mathematical formulation

### 2.1 Input space

- **Raw input:** A clip of **T** frames, each an RGB image.  
  - Frames: $I_1, I_2, \ldots, I_T$, each $I_t \in \mathbb{R}^{H \times W \times 3}$.  
  - After stacking and preprocessing we get a tensor

$$
\mathbf{X} \in \mathbb{R}^{1 \times T \times 3 \times H_c \times W_c}
$$

  with $H_c = W_c = 256$ (crop size), and batch size 1.

### 2.2 Patchification (tubelets)

- The model splits the clip into **spatiotemporal patches** (tubelets):
  - **Spatial:** each frame is divided into $P \times P$ patches (e.g. $P = 16$ → $(256/16)^2 = 256$ patches per frame).  
  - **Temporal:** often $\tau$ consecutive frames are grouped (e.g. $\tau = 2$).  
- Each tubelet is a small “cube” of shape $\tau \times P \times P \times 3$. It is linearly projected to a **token** in $\mathbb{R}^D$:

$$
\mathbf{z}_i = \mathbf{E} \, \mathrm{vec}(\mathrm{Tubelet}_i) + \mathbf{e}_{\mathrm{pos},i}, \qquad i = 1,\ldots,N.
$$

  - $\mathbf{E}$: patch embedding matrix.  
  - $\mathbf{e}_{\mathrm{pos},i}$: positional embedding for token $i$.  
  - **N** = total number of tokens (e.g. $N = (T/\tau) \times (256/P)^2$ or similar depending on exact config).

So the clip is represented as a **sequence of N vectors** in $\mathbb{R}^D$:

$$
\mathbf{Z} = (\mathbf{z}_1, \ldots, \mathbf{z}_N) \in \mathbb{R}^{N \times D}.
$$

### 2.3 Transformer encoder

- The encoder is a **Transformer** (self-attention + MLP layers). It takes the token sequence and updates it:

$$
\mathbf{Z}^{(0)} = \mathbf{Z}, \qquad
\mathbf{Z}^{(\ell+1)} = \mathrm{TransformerLayer}^{(\ell)}\bigl(\mathbf{Z}^{(\ell)}\bigr), \quad \ell = 0,\ldots,L-1.
$$

- Each layer typically does:
  - **Self-attention:** each token attends to all others and is updated.  
  - **MLP:** a 2-layer feed-forward on each token.  
  - Residual connections and layer norm.

- **Output:** the last layer’s sequence:

$$
\mathbf{H} = \mathbf{Z}^{(L)} \in \mathbb{R}^{N \times D}.
$$

  This is exactly what you get as **`last_hidden_state`**: one D-dimensional vector per token.

### 2.4 What we use for navigation (conceptually)

- For a **single embedding per clip** (e.g. for a policy), we reduce over the token dimension, for example:

$$
\mathbf{h}_{\mathrm{clip}} = \frac{1}{N}\sum_{i=1}^{N} \mathbf{H}_i \quad \in \mathbb{R}^D,
$$

  or take a dedicated “[CLS]” token if the model provides one.  
- So the **map** from your 16-frame Replica images to one vector is:

$$
(I_1,\ldots,I_T) \;\xrightarrow{\mathrm{preprocess}}\; \mathbf{X} \;\xrightarrow{\mathrm{patchify}}\; \mathbf{Z} \;\xrightarrow{\mathrm{Transformer}}\; \mathbf{H} \;\xrightarrow{\mathrm{pool}}\; \mathbf{h}_{\mathrm{clip}}.
$$

---

## 3. Summary table

| Stage        | Input shape (conceptually)     | Output shape (conceptually) |
|-------------|---------------------------------|-----------------------------|
| Load & stack | 4 images (H×W×3)               | (T, 3, H, W), T=16          |
| Processor    | (T, 3, H, W)                    | (1, T, 3, 256, 256)         |
| Patchify     | (1, T, 3, 256, 256)             | (1, N, D) tokens            |
| Encoder      | (1, N, D)                       | (1, N, D) = `last_hidden_state` |
| Pool (later) | (1, N, D)                       | (1, D) one vector per clip  |

**Actual run** (16 frames, Replica assets): encoder output shape **(1, 2048, 1024)** → $N = 2048$ tokens, $D = 1024$ (ViT-L hidden size). Mean ≈ 0.01, std ≈ 3.23.

So: **the command runs the first four stages and gives you the encoder output (1, N, D).** The math above describes how the Replica asset images are turned into that embedding sequence.
