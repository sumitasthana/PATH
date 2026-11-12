"""
Stream-side plasticity metrics (measurement only; the training run is unchanged, see the Gate 0 fingerprint test).

Enabled with --ap_metrics 1. Recorded per task in the result file, next to accuracy, dW and the fork probes.

Per update (every --ap_lp_every steps), on the training batch just used:
  loss progress (Elsayed and Mahmood 2024): (L_before - L_after) / L_before, the drop in the batch's own loss produced
  by the update. Low progress at matched learning rate = the model has become harder to move.

At every task boundary t (t = 0 before task 1, then after each task), on fixed images (the first --ap_metric_n test
images of the current task; at t = 0 those of task 1), under a full random-number guard and with no parameter change:
  feature effective rank (Roy and Vetterli 2007) and srank, delta = 0.01 (Kumar et al. 2021), of the CLS features;
  dormant MLP units (Sokar et al. 2023): fraction of MLP hidden units in all 12 blocks whose normalised mean |activation|
      is <= tau, tau in {0.01, 0.025, 0.1};
  adapter parameter norm (Lewandowski et al., Kumar et al. weight-norm growth; for the full model, the norm of the
      measured attention matrices) and the increment
      dW_step(t) = ||W_eff(t) - W_eff(t-1)||_F / ||W0||_F, how far the effective weights moved during task t;
  gradient alignment (Lyle et al. 2023): norm of the mean adapter gradient and mean pairwise cosine between
      per-example adapter gradients on --ap_grad_n examples, under the training loss (local cross-entropy);
  sharpness: both ends of the Hessian spectrum of the same loss in adapter space (--ap_hess_n examples), by power
      iteration on Hessian-vector products (--ap_hess_iters iterations), then on the shifted operator H - lam_1 I.
      lambda_max (most positive eigenvalue) is the sharpness; lambda_min shows negative curvature; each is stored
      with its relative residual ||Hv - lam v|| / |lam| so unconverged values can be excluded. The raw values are in
      adapter-parameter units, so they are compared within one adapter type, not between LoRA and OFT.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from models.ap_utils.plasticity import RNGGuard

TAUS = (0.01, 0.025, 0.1)


def add_metric_args(parser):
    g = parser.add_argument_group('Stream plasticity metrics')
    g.add_argument('--ap_metrics', type=int, default=0, choices=[0, 1])
    g.add_argument('--ap_lp_every', type=int, default=10, help='Measure loss progress every k training steps.')
    g.add_argument('--ap_metric_n', type=int, default=256, help='Images for rank and dormant-unit metrics.')
    g.add_argument('--ap_grad_n', type=int, default=32, help='Examples for per-example gradient alignment.')
    g.add_argument('--ap_hess_n', type=int, default=64, help='Examples for the Hessian power iteration.')
    g.add_argument('--ap_hess_iters', type=int, default=20)
    return parser


def effective_rank(f):
    f = f.double() - f.double().mean(0, keepdim=True)
    s = torch.linalg.svdvals(f)
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def srank(f, delta=0.01):
    s = torch.linalg.svdvals(f.double())
    c = torch.cumsum(s, 0) / s.sum()
    return int((c < 1 - delta).sum().item()) + 1


def _dot(a, b):
    return sum((x * y).sum().item() for x, y in zip(a, b))


def _power(hvp, shapes_like, iters, shift=0.0):
    """Power iteration on (H - shift I). Returns (Rayleigh quotient of H, relative residual ||Hv - lam v|| / |lam|)."""
    v = [torch.randn_like(p) for p in shapes_like]
    nv = math.sqrt(_dot(v, v))
    v = [z / nv for z in v]
    for _ in range(iters):
        hv = hvp(v)
        w = [h - shift * z for h, z in zip(hv, v)]
        nw = math.sqrt(_dot(w, w))
        v = [z.detach() / max(nw, 1e-30) for z in w]
    hv = hvp(v)
    lam = _dot(hv, v)
    r = math.sqrt(sum(((h - lam * z) ** 2).sum().item() for h, z in zip(hv, v)))
    return float(lam), float(r / max(abs(lam), 1e-30))


def hessian_extremes(loss, params, iters):
    """Both ends of the Hessian spectrum of `loss` w.r.t. `params`, from Hessian-vector products.

    Step 1: power iteration on H gives lam_1, the eigenvalue of largest magnitude (either sign).
    Step 2: power iteration on H - lam_1 I converges to the eigenvalue at the other end of the spectrum.
    lambda_max (sharpness, the most positive eigenvalue) and lambda_min are the larger and smaller of the two.
    Each comes with its relative residual ||Hv - lam v|| / |lam| as a convergence check (0 = exact eigenpair).
    """
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    pairs = [(g, p) for g, p in zip(grads, params) if g is not None]
    gs, ps = [g for g, _ in pairs], [p for _, p in pairs]

    def hvp(v):
        return [h.detach() for h in torch.autograd.grad(gs, ps, grad_outputs=v, retain_graph=True)]

    lam1, r1 = _power(hvp, ps, iters)
    lam2, r2 = _power(hvp, ps, iters, shift=lam1)
    (lmax, rmax), (lmin, rmin) = sorted([(lam1, r1), (lam2, r2)], reverse=True)
    return dict(top=lam1, top_resid=r1, lambda_max=lmax, lambda_max_resid=rmax, lambda_min=lmin, lambda_min_resid=rmin)


def top_hessian_eig(loss, params, iters, seed_vec=None):
    """Eigenvalue of largest magnitude (kept for continuity with batch A's `hessian_top_eig`)."""
    return hessian_extremes(loss, params, iters)['top']


def _mlp_acts(net):
    """Modules whose outputs are the MLP hidden activations (timm Mlp.act) of every block."""
    return [m.act for m in net.body.modules() if type(m).__name__ == 'Mlp' and hasattr(m, 'act')]


class StreamMetrics:
    def __init__(self, model):
        self.m = model
        self.args = model.args
        self.prev_eff = None          # W_eff at the previous boundary (CPU float32), for dW_step
        self.lp = []                  # per-update loss progress within the current task

    # ------------------------------------------------------------------ per update
    @torch.no_grad()
    def after_step(self, inputs, labels, past, seen, loss_before, step):
        if step % self.args.ap_lp_every != 0:
            return
        out = self.m.net(inputs)[:, past:seen]
        after = F.cross_entropy(out, labels - past).item()
        self.lp.append((loss_before, after))

    def take_loss_progress(self):
        if not self.lp:
            return {}
        b = np.array([x[0] for x in self.lp])
        a = np.array([x[1] for x in self.lp])
        rel = (b - a) / np.maximum(b, 1e-12)
        self.lp = []
        return dict(lp_rel_mean=float(rel.mean()), lp_rel_median=float(np.median(rel)),
                    lp_abs_mean=float((b - a).mean()), lp_n=int(len(rel)), lp_loss_before_mean=float(b.mean()))

    # ------------------------------------------------------------------ at a boundary
    def _images(self, dataset, t, n):
        loaders = dataset.test_loaders
        loader = loaders[min(max(t, 1) - 1, len(loaders) - 1)]      # joint training: one loader, all classes
        xs, ys = [], []
        for batch in loader:
            xs.append(batch[0])
            ys.append(batch[1])
            if sum(len(x) for x in xs) >= n:
                break
        return torch.cat(xs)[:n].to(self.m.device), torch.cat(ys)[:n].to(self.m.device)

    def _loss(self, x, y, t):
        past, seen = self.m.offsets_for_task(max(t, 1) - 1)
        return F.cross_entropy(self.m.net(x)[:, past:seen], y - past), past, seen

    def measure(self, dataset, t):
        net = self.m.net
        params = net.metric_parameters() if hasattr(net, 'metric_parameters') else net.adapter_parameters()
        was = net.training
        out = {}
        with RNGGuard():
            net.eval()
            x, y = self._images(dataset, t, self.args.ap_metric_n)
            # features, ranks, dormant units
            acts = _mlp_acts(net)
            sums, hooks = [None] * len(acts), []
            for i, a in enumerate(acts):
                def hook(_m, _i, o, i=i):
                    v = o.detach().abs().float().mean(dim=tuple(range(o.dim() - 1)))
                    sums[i] = v if sums[i] is None else sums[i] + v
                hooks.append(a.register_forward_hook(hook))
            with torch.no_grad():
                feats = torch.cat([net(x[i:i + 64], returnt='features').float() for i in range(0, len(x), 64)])
            for h in hooks:
                h.remove()
            out['erank'] = effective_rank(feats)
            out['srank'] = srank(feats)
            if sums and sums[0] is not None:
                for tau in TAUS:
                    frac = [float(((s / s.mean()) <= tau).float().mean()) for s in sums]
                    out[f'dormant_{tau:g}'] = float(np.mean(frac))
                    out[f'dormant_{tau:g}_per_block'] = [round(v, 4) for v in frac]
            # weight norms and per-task movement
            with torch.no_grad():
                out['adapter_norm'] = float(math.sqrt(sum(p.double().pow(2).sum().item() for p in params)))
                eff = [(d + W).float().cpu() for d, W in net.weight_deltas()]
                den = math.sqrt(sum(W.double().pow(2).sum().item() for _, W in net.weight_deltas()))
                if self.prev_eff is not None:
                    num = math.sqrt(sum((a - b).double().pow(2).sum().item() for a, b in zip(eff, self.prev_eff)))
                    out['dW_step'] = num / den
                self.prev_eff = eff
            # per-example gradient alignment
            g = []
            for i in range(min(self.args.ap_grad_n, len(x))):
                loss, _, _ = self._loss(x[i:i + 1], y[i:i + 1], t)
                gr = torch.autograd.grad(loss, params, allow_unused=True)
                g.append(torch.cat([(z if z is not None else torch.zeros_like(p)).flatten() for z, p in zip(gr, params)]))
            G = torch.stack(g).float()
            out['grad_norm_mean'] = float(G.mean(0).norm())
            out['grad_norm_per_example'] = float(G.norm(dim=1).mean())
            Gn = G / G.norm(dim=1, keepdim=True).clamp_min(1e-12)
            C = Gn @ Gn.t()
            k = C.shape[0]
            off = C[~torch.eye(k, dtype=torch.bool, device=C.device)]
            out['grad_cos_mean'] = float(off.mean())
            out['grad_cos_neg_frac'] = float((off < 0).float().mean())
            del G, Gn, C, g
            # sharpness: top Hessian eigenvalue in adapter space (power iteration)
            xh, yh = x[:self.args.ap_hess_n], y[:self.args.ap_hess_n]
            # second derivatives need the plain attention kernel (fused SDPA kernels have no double backward)
            from torch.nn.attention import sdpa_kernel, SDPBackend
            with sdpa_kernel([SDPBackend.MATH]):
                loss, _, _ = self._loss(xh, yh, t)
                h = hessian_extremes(loss, params, self.args.ap_hess_iters)
            out['hessian_top_eig'] = h['top']                       # largest magnitude, either sign (batch A key)
            out['hessian_top_resid'] = h['top_resid']
            out['hessian_lambda_max'] = h['lambda_max']             # sharpness
            out['hessian_lambda_max_resid'] = h['lambda_max_resid']
            out['hessian_lambda_min'] = h['lambda_min']
            out['hessian_lambda_min_resid'] = h['lambda_min_resid']
        for p in params:
            p.grad = None
        net.zero_grad(set_to_none=True)
        net.train(was)
        if str(self.m.device).startswith('cuda'):
            torch.cuda.empty_cache()
        return out
