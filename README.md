# 🎧 HTDemucs Implementation (Demucs v3)

This project is a clean PyTorch implementation of **HTDemucs (Hybrid Transformer Demucs)**, based on the official Facebook Demucs v3 paper. The goal is to build the model architecture from scratch following the paper's logic, focusing on clarity and modularity.

---

## ✅ Current Progress

### 1. Project Setup

- Python virtual environment (`venv`)
- Installed: `torch`, `torchaudio`, `librosa`, `tqdm`, `pyyaml`
- Directory structure:
  ```
  demucs/
  ├── src/
  │   └── model/
  │       └── htdemucs.py
  ├── notebook/
  │   └── model_test.ipynb
  ├── config/
  │   └── default.yaml
  ├── requirements.txt
  └── README.md
  ```

---

### 2. Input Shape Convention

All audio tensors follow the shape:

```
(B, C, T)
```

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size |
| `C` | Number of channels (1 for mono) |
| `T` | Time steps (e.g., 16000 for 1 second of 16kHz audio) |

---

### 3. EncoderBlock

A single downsampling block that performs:

- `Conv1D`: Feature extraction and downsampling
- `GELU`: Non-linearity (smooth alternative to ReLU)
- `LayerNorm`: Channel-wise normalization for stable training

---

### 4. HTDemucs Model (Encoder + Transformer + Decoder)

The full model is composed of:

1. **Encoder Stack**: Conv1D blocks that extract and compress temporal features.
2. **Transformer Block**: Captures long-term dependencies with self-attention.
3. **Decoder Stack**: ConvTranspose1D blocks that upsample the features and restore resolution.

Output shape is progressively reduced in the encoder (stride=4), passed through a Transformer, and then restored in the decoder. Skip connections between encoder and decoder layers are used to improve reconstruction accuracy.

---

### 5. Transformer Block

The bottleneck of HTDemucs uses a stack of Transformer layers to capture long-term dependencies in the encoded feature sequence. This includes:

- **Learnable positional encoding** to preserve temporal order
- **Multi-head self-attention and feed-forward layers** for global context modeling
- Input/output shape transformed between `(B, C, T)` and `(B, T, C)` for compatibility

Configuration:
- `d_model`: 1536
- `nhead`: 8
- `num_layers`: 6
- `dropout`: 0.1

---

### 6. Decoder Block

The decoder progressively upsamples the temporal dimension using `ConvTranspose1D`, reducing channel size at each stage. It mirrors the encoder structure in reverse.

Each block includes:
- `ConvTranspose1D` for upsampling
- `GELU` activation
- `LayerNorm` for channel-wise normalization

Decoder output shapes are adjusted to match the original input resolution, typically with skip connections added later.

---

### 7. Skip Connections

To enhance signal reconstruction, skip connections are added between each encoder and decoder layer (in reverse order).  
This enables the decoder to reuse high-resolution features from the encoder.

Time resolution mismatches between encoder and decoder features are handled via cropping or linear interpolation.

---

### 8. Debugging Notes

If changes to `htdemucs.py` do not apply in notebooks, use:

```python
import importlib
import model.htdemucs
importlib.reload(model.htdemucs)
```

This ensures the latest version is reloaded without restarting the kernel.

---

## 🔜 Upcoming Modules

- [ ] **GELU vs ReLU** explanation
- [ ] **Conv1D vs ConvTranspose1D**
- [ ] **TransformerBlock** (Multi-head Attention + FFN)
- [ ] **Learnable Positional Encoding**
- [ ] **DecoderBlock** with upsampling
- [ ] **Skip Connections** between encoder and decoder
- [ ] **Final Output Projection**
- [ ] **Loss Function & Training Loop**
- [ ] **Dataset Loader** (e.g., MUSDB18)
- [ ] **Evaluation & Inference**
- [ ] **Config Parameter Integration**

---

## 📚 Reference

- [Demucs v3 paper (HTDemucs)](https://arxiv.org/abs/2211.08553)
- [Official Demucs GitHub](https://github.com/facebookresearch/demucs)
