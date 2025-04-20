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

### 4. HTDemucs (Encoder Only)

- Encoder expands channels progressively
- Time resolution decreases due to stride=4
- Output shape example: `(1, 1536, ~15)` for input of `(1, 1, 16000)`

---

### 5. Debugging Notes

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

---

## 📚 Reference

- [Demucs v3 paper (HTDemucs)](https://arxiv.org/abs/2211.08553)
- [Official Demucs GitHub](https://github.com/facebookresearch/demucs)
