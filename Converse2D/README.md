### Kernel Registry

---

**Kernel Tree**

```
├---v1 Translation from python to CPP
├---v2 Add FB/F2B cache & broadcast replace repeat
```

**Tested Device**

- NVIDIA RTX 2080ti
- NVIDIA RTX 4090
- NVIDIA RTX 5060ti 16g

**Installation**

```python
cd ./Converse2D
pip install . --no-build-isolation
```

**Usage**

```python
import torch
import torch_converse2d

out = torch.ops.converse2d.forward(x, x0, weight, bias, scale, eps)
print(torch.ops.converse2d)
```
