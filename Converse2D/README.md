### Kernel Registry
----------
**Kernel Details**
We offer four versions Converse2d Kernel.
- v1:  Translation from python to CPP **faster**
- v2:   Add FB/F2B cache & broadcast replace repeat **much faster**
- v3:  `​splits→permute→view→mean` to `block mean CUDA kernel` **fastest**
- v4:   STy s-fold upsampler CUDA kernel **fastest**

**Tested Device**
- NVIDIA RTX 2080ti
- NVIDIA RTX 4090
- NVIDIA RTX 5060ti 16g

Under different circumstances, **v3** and **v4** each have their own performance advantages, but they are both faster than **v1** and **v2**.

We highly recommend you to run `test/test_speed.py` first to choose the most suitable backend for GPU.


**Installation**

```python
cd ./Converse2D
# Remember to choose the wanted kernel version
CONVERSE2D_VARIANT={v1,v2,v3,v4} pip install -e .
```

**Usage**

```python
import torch
import torch_converse2d

out = torch.ops.converse2d.forward(x, x0, weight, bias, scale, eps)
print(torch.ops.converse2d)
```

**TODO**
- [ ] Temporary Tensor Reuse and In-Place Writing
- [ ] Larger batched FFT
- [ ] Eliminate redundant calculations of `conj/abs/pow(2)`
- [ ] The minimal necessary policy for `contiguous()`
- [ ] R2C/C2R (Real FFT) replaces C2C **(Optional)**
- [ ] Mixed precision **(Optional)**
- [ ] Adaptive padding **(Optional)**