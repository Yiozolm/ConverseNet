"""Ordinary eager FP32 training; no CUDA Graph integration."""
import time
import torch


class TrainingRunner:
    def __init__(self, model, batches, targets, lr=1e-4, momentum=.9, *, warmup_steps=3):
        if torch.is_inference_mode_enabled():
            raise ValueError('training cannot run in inference mode')
        if not batches or len(batches)!=len(targets) or warmup_steps<1:
            raise ValueError('nonempty matching batches/targets and positive warmup required')
        self.model=model.train()
        self.batches,self.targets=batches,targets
        self._params=dict(model.named_parameters())
        self._buffers=dict(model.named_buffers())
        self.device=next(iter(self._params.values())).device
        assert self.device.type=='cuda'
        assert all(p.device==self.device and p.dtype==torch.float32 for p in self._params.values())
        self._initial_params={n:p.detach().clone() for n,p in self._params.items()}
        self._initial_buffers={n:t.detach().clone() for n,t in self._buffers.items()}
        self.optimizer=torch.optim.SGD(model.parameters(),lr=lr,momentum=momentum,
                                       dampening=0,weight_decay=0,foreach=False,fused=False)
        for p in self._params.values():
            if momentum:self.optimizer.state[p]['momentum_buffer']=torch.zeros_like(p)
        self.loss=torch.zeros((),device=self.device,dtype=torch.float32)
        torch.cuda.synchronize()
        started=time.perf_counter()
        for _ in range(warmup_steps):self.step()
        torch.cuda.synchronize()
        self.warmup_ms=(time.perf_counter()-started)*1000
        self.reset()

    @torch.no_grad()
    def reset(self):
        for n,p in self._params.items():
            p.copy_(self._initial_params[n])
            p.grad=None
            buffer=self.optimizer.state[p].get('momentum_buffer')
            if buffer is not None:buffer.zero_()
        for n,t in self._buffers.items():t.copy_(self._initial_buffers[n])
        self.loss=torch.zeros((),device=self.device,dtype=torch.float32)

    def step(self,batches=None,targets=None):
        batches=self.batches if batches is None else batches
        targets=self.targets if targets is None else targets
        if not batches or len(batches)!=len(targets):
            raise ValueError('matching batches and targets required')
        self.optimizer.zero_grad(set_to_none=True)
        losses=[]
        with torch.enable_grad():
            for batch,target in zip(batches,targets):
                output=self.model(*batch)
                assert output.shape==target.shape
                loss=(output-target).square().mean()/len(batches)
                loss.backward()
                losses.append(loss.detach())
                del output,loss
            self.optimizer.step()
        self.loss=losses[0] if len(losses)==1 else torch.stack(losses).sum()
        return self.loss

    @torch.no_grad()
    def snapshot(self):
        return dict(loss=self.loss.clone(),
                    params={n:p.detach().clone() for n,p in self._params.items()},
                    grads={n:p.grad.detach().clone() for n,p in self._params.items() if p.grad is not None},
                    momentum={n:self.optimizer.state[p]['momentum_buffer'].clone()
                              for n,p in self._params.items() if 'momentum_buffer' in self.optimizer.state[p]},
                    buffers={n:t.clone() for n,t in self._buffers.items()})
