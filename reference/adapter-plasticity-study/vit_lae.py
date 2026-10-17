"""
Frozen timm ViT-B/16 + one adapter + linear head. One backbone for every preset, from the LAE reproduction
(gate 3) to the study protocol, so each protocol step changes exactly one argument.

--ap_adapter lae_kvlora (default; LAE, Gao et al. ICCV 2023, libml/model/pet/lora.py KVLoRA):
    rank-5 LoRA on the K and V rows of attn.qkv only (Q untouched), A ~ kaiming_uniform(a=sqrt(5)), B = 0,
    one learnable scalar per K and per V initialised to 1.0, blocks --ap_blocks (LAE: 0-4).
--ap_adapter lora: PEFT LoRA (r=--ap_lora_r, alpha=--ap_lora_alpha, no dropout) on --ap_targets of --ap_blocks.
--ap_adapter oft:  PEFT OFT, exact Cayley (no Neumann series), block size --ap_oft_block_size, same placement.

Backbone: --ap_timm_model, frozen, no pre-logits, feature = CLS token after the final LayerNorm.
Head: linear, weights N(0, 0.001^2), bias 0 (LAE SimpleHead.weights_init). The 100-way head exists from the
start; with local cross-entropy only the current task's rows receive gradient, the same computation as LAE's
per-task classifiers.

weight_deltas() yields (W_eff - W0, W0) for every adapted matrix, the input to the relative weight change dW.

Registered as `vit-lae`.
"""
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from backbone import MammothBackbone, register_backbone


class KVLoRAQKV(nn.Module):
    """Wraps a frozen qkv nn.Linear; adds s_k B_k A_k to the K rows and s_v B_v A_v to the V rows."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        d = base.in_features
        assert base.out_features == 3 * d
        self.d = d
        self.lora_A = nn.ParameterList([nn.Parameter(torch.zeros(rank, d)) for _ in range(2)])
        self.lora_B = nn.ParameterList([nn.Parameter(torch.zeros(d, rank)) for _ in range(2)])
        self.scale = nn.ParameterList([nn.Parameter(torch.tensor(1.0)) for _ in range(2)])
        for i in range(2):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])

    def delta(self):
        kv = torch.cat([s * (B @ A) for s, A, B in zip(self.scale, self.lora_A, self.lora_B)], dim=0)
        return torch.cat([torch.zeros(self.d, self.d, device=kv.device, dtype=kv.dtype), kv], dim=0)

    def forward(self, x):
        return F.linear(x, self.base.weight + self.delta(), self.base.bias)


def expected_adapter_params(adapter, blocks, targets, d, lora_r, kv_rank, oft_block):
    """Closed-form trainable-parameter count for ViT width d (qkv: d -> 3d, proj: d -> d)."""
    nb = len(blocks)
    shapes = {'qkv': (d, 3 * d), 'proj': (d, d)}
    if adapter == 'lae_kvlora':
        return nb * (4 * kv_rank * d + 2)
    if adapter == 'lora':
        return nb * sum(lora_r * (shapes[t][0] + shapes[t][1]) for t in targets)
    if adapter == 'oft':
        return nb * sum((shapes[t][0] // oft_block) * oft_block * (oft_block - 1) // 2 for t in targets)
    raise ValueError(adapter)


@register_backbone("vit-lae")
class VitLAE(MammothBackbone):
    def __init__(self, num_classes: int, ap_adapter: str = 'lae_kvlora', ap_blocks: str = '0,1,2,3,4',
                 ap_targets: str = 'qkv', ap_lae_rank: int = 5, ap_lora_r: int = 8, ap_lora_alpha: float = 8.0,
                 ap_oft_block_size: int = 48, ap_timm_model: str = 'vit_base_patch16_224.orig_in21k',
                 ap_pretrained: int = 1):
        super().__init__()
        assert ap_adapter in ('lae_kvlora', 'lora', 'oft'), ap_adapter
        img = 224
        if os.environ.get('AP_FAKE_DATA') == '1':          # CPU tests: tiny model, no download
            ap_timm_model, ap_pretrained, img = 'vit_tiny_patch16_224', 0, 32
        vit = timm.create_model(ap_timm_model, pretrained=bool(ap_pretrained), num_classes=0,
                                global_pool='token', img_size=img)
        for p in vit.parameters():
            p.requires_grad_(False)
        self.adapter = ap_adapter
        self.blocks_adapted = [int(b) for b in str(ap_blocks).split(',') if b.strip() != '']
        self.targets = [t.strip() for t in str(ap_targets).split(',') if t.strip() != '']
        assert set(self.targets) <= {'qkv', 'proj'}, self.targets
        self.num_features = vit.num_features
        self.expected_params = expected_adapter_params(ap_adapter, self.blocks_adapted, self.targets,
                                                       vit.num_features, ap_lora_r, ap_lae_rank, ap_oft_block_size)
        if ap_adapter == 'lae_kvlora':
            assert self.targets == ['qkv'], 'LAE KVLoRA adapts attn.qkv only'
            for b in self.blocks_adapted:
                vit.blocks[b].attn.qkv = KVLoRAQKV(vit.blocks[b].attn.qkv, ap_lae_rank)
            self.body = vit
        else:
            from peft import LoraConfig, OFTConfig, get_peft_model
            regex = r'.*blocks\.(%s)\.attn\.(%s)' % ('|'.join(map(str, self.blocks_adapted)), '|'.join(self.targets))
            if ap_adapter == 'lora':
                cfg = LoraConfig(r=ap_lora_r, lora_alpha=ap_lora_alpha, lora_dropout=0.0, target_modules=regex)
            else:
                cfg = OFTConfig(r=0, oft_block_size=ap_oft_block_size, target_modules=regex, module_dropout=0.0,
                                use_cayley_neumann=False)
            self.body = get_peft_model(vit, cfg)
        self.head = nn.Linear(self.num_features, num_classes)
        nn.init.normal_(self.head.weight, std=0.001)
        nn.init.constant_(self.head.bias, 0.0)
        self.pretrained_cfg = vit.pretrained_cfg

    def adapter_parameters(self):
        return [p for p in self.body.parameters() if p.requires_grad]

    def head_parameters(self):
        return list(self.head.parameters())

    def base_named_parameters(self):
        """Frozen pretrained parameters only (adapter parameters excluded)."""
        ad = {id(p) for p in self.adapter_parameters()}
        return [(n, p) for n, p in self.body.named_parameters() if id(p) not in ad]

    @torch.no_grad()
    def weight_deltas(self):
        """(W_eff - W0, W0) in float64 for every adapted weight matrix."""
        if self.adapter == 'lae_kvlora':
            for m in self.body.modules():
                if isinstance(m, KVLoRAQKV):
                    yield m.delta().double(), m.base.weight.double()
            return
        import peft.tuners.oft.layer as oft_layer
        import peft.tuners.lora.layer as lora_layer
        for m in self.body.modules():
            if self.adapter == 'oft' and isinstance(m, oft_layer.OFTLayer) and 'default' in getattr(m, 'oft_R', {}):
                W = m.get_base_layer().weight.double()
                rot = m.oft_R['default']
                R = torch.block_diag(*rot._cayley_batch(rot.weight, rot.block_size, rot.use_cayley_neumann,
                                                        rot.num_cayley_neumann_terms).double())
                yield W @ R.t() - W, W
            elif self.adapter == 'lora' and isinstance(m, lora_layer.LoraLayer) and 'default' in getattr(m, 'lora_A', {}):
                W = m.get_base_layer().weight.double()
                A, B = m.lora_A['default'].weight.double(), m.lora_B['default'].weight.double()
                yield m.scaling['default'] * (B @ A), W

    def forward(self, x: torch.Tensor, returnt='out'):
        assert returnt in ('out', 'features', 'both', 'full')
        feats = self.body(x)          # CLS token after final norm (global_pool='token', num_classes=0)
        if returnt == 'features':
            return feats
        out = self.head(feats)
        if returnt == 'out':
            return out
        return out, feats
