from .common_utils_mamba import *
# custom_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import math
from functools import partial
import copy


__all__ = ("VSSBlock", "SimpleStem", "VisionClueMerge", "XSSBlock", "OnlineDPCClus", "SEIN", "SEINBlock", "VSSBlock_DSM", "SS2D_DSM")



class SS2D(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        y = self.forward_core(x, channel_first=(self.d_conv > 1))
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out



class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        if hidden_features is None:
            hidden_features = in_features
        
        groups = max(1, hidden_features)
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x



class XSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            n: int = 1,
            mlp_ratio=4.0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.in_proj = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        ) if in_channels != hidden_dim else nn.Identity()
        self.hidden_dim = hidden_dim
        # ==========SSM============================
        self.norm = norm_layer(hidden_dim)
        self.ss2d = nn.Sequential(*(SS2D(d_model=self.hidden_dim,
                                         d_state=ssm_d_state,
                                         ssm_ratio=ssm_ratio,
                                         ssm_rank_ratio=ssm_rank_ratio,
                                         dt_rank=ssm_dt_rank,
                                         act_layer=ssm_act_layer,
                                         d_conv=ssm_conv,
                                         conv_bias=ssm_conv_bias,
                                         dropout=ssm_drop_rate, ) for _ in range(n)))
        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate)

    def forward(self, input):
        input = self.in_proj(input)
        # ====================
        X1 = self.lsblock(input)
        input = input + self.drop_path(self.ss2d(self.norm(X1)))
        # ===================
        if self.mlp_branch:
            input = input + self.drop_path(self.mlp(self.norm2(input)))
        return input



class VSSBlock(nn.Module):
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                # ==========================
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                # ==========================
                dropout=ssm_drop_rate,
                # bias=False,
                # ==========================
                # dt_min=0.001,
                # dt_max=0.1,
                # dt_init="random",
                # dt_scale="random",
                # dt_init_floor=1e-4,
                initialize=ssm_init,
                # ==========================
                forward_type=forward_type,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        x = input + self.drop_path(self.op(self.norm(X1)))
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x



class SimpleStem(nn.Module):
    def __init__(self, inp, embed_dim, ks=3):
        super().__init__()
        self.hidden_dims = embed_dim // 2
        self.conv = nn.Sequential(
            nn.Conv2d(inp, self.hidden_dims, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(self.hidden_dims),
            nn.GELU(),
            nn.Conv2d(self.hidden_dims, embed_dim, kernel_size=ks, stride=2, padding=autopad(ks, d=1), bias=False),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.conv(x)



class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.hidden = int(dim * 4)

        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        y = torch.cat([
            x[..., ::2, ::2],
            x[..., 1::2, ::2],
            x[..., ::2, 1::2],
            x[..., 1::2, 1::2]
        ], dim=1)
        return self.pw_linear(y)



class OnlineDPCClus(nn.Module):
    """
    Online Density Peak Clustering (ODPC) module with k-NN adaptive bandwidth
    Maintains a streaming memory bank of sampled features and incrementally performs density peak estimation
    """
    def __init__(self, 
                 feature_dim: int = 256,
                 num_clusters: int = 8,
                 k_neighbors: int = 20, 
                 alpha: float = 1.0,     
                 memory_size: int = 1000,
                 update_freq: int = 10,
                 temperature: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_clusters = num_clusters
        self.k_neighbors = k_neighbors
        self.alpha = alpha
        self.memory_size = memory_size
        self.update_freq = update_freq
        self.temperature = temperature
        
        self.feature_projector = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.register_buffer('memory_bank', torch.randn(memory_size, feature_dim))
        self.register_buffer('memory_ptr', torch.zeros(1, dtype=torch.long))
        self.register_buffer('update_counter', torch.zeros(1, dtype=torch.long))
        self.register_buffer('cluster_centers', torch.randn(num_clusters, feature_dim))
        self.register_buffer('cluster_densities', torch.zeros(num_clusters))
        self.register_buffer('num_active_clusters', torch.zeros(1, dtype=torch.long))
        self.register_buffer('zero_tensor', torch.zeros(1, 1, 1, 1))
        self.register_buffer('temp_centers', torch.zeros(num_clusters, feature_dim))
        self.register_buffer('temp_densities', torch.zeros(num_clusters))
    
    def __deepcopy__(self, memo):

        result = self.__class__(
            feature_dim=self.feature_dim,
            num_clusters=self.num_clusters,
            k_neighbors=self.k_neighbors,
            alpha=self.alpha,
            memory_size=self.memory_size,
            update_freq=self.update_freq,
            temperature=self.temperature
        )
        memo[id(self)] = result
        
        if hasattr(self, 'memory_bank') and self.memory_bank.is_cuda:
            result = result.cuda()
        
        return result
    
    def update_memory_bank(self, features):
        """Update memory bank with new features in circular fashion"""
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(0, 2, 1).contiguous().view(-1, C)
        
        # Random sampling
        num_samples = min(features_flat.size(0), self.memory_size // 4)
        if num_samples > 0:
            indices = torch.randperm(features_flat.size(0))[:num_samples]
            sampled_features = features_flat[indices]
            
            with torch.no_grad():
                # Circular update
                ptr = self.memory_ptr.item()
                end_ptr = min(ptr + num_samples, self.memory_size)
                actual_samples = end_ptr - ptr
                
                self.memory_bank[ptr:end_ptr].copy_(sampled_features[:actual_samples])
                
                if end_ptr == self.memory_size:
                    remaining = num_samples - actual_samples
                    if remaining > 0:
                        self.memory_bank[:remaining].copy_(sampled_features[actual_samples:actual_samples+remaining])
                        self.memory_ptr.fill_(remaining)
                    else:
                        self.memory_ptr.fill_(0)
                else:
                    self.memory_ptr.add_(actual_samples)
    
    def compute_density_knn(self, features, memory_bank):
        """
        Compute local density using k-NN adaptive bandwidth 
        """
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, HW, C]
        
        k = min(self.k_neighbors, memory_bank.size(0))
        
        distances = torch.cdist(features_flat, memory_bank)  # [B, HW, M]
        knn_distances, _ = torch.topk(distances, k, dim=-1, largest=False)  # [B, HW, k]
        r_k = knn_distances[..., -1]  # [B, HW] 
        adaptive_bandwidth = self.alpha * r_k  # [B, HW]
        adaptive_bandwidth = torch.clamp(adaptive_bandwidth, min=1e-8)
        adaptive_bandwidth_expanded = adaptive_bandwidth.unsqueeze(-1)  # [B, HW, 1]
        
        weights = torch.exp(-(distances / adaptive_bandwidth_expanded) ** 2)  # [B, HW, M]
        density = weights.sum(dim=-1)  # [B, HW]
        
        return density.view(B, 1, H, W)
    
    def find_density_peaks(self, features, density_map):
        """Find top-K density peaks as cluster centers"""
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, HW, C]
        density_flat = density_map.view(B, -1)  # [B, HW]
        
        with torch.no_grad():
            self.temp_centers.zero_()
            self.temp_densities.zero_()
            
            for b in range(B):
                # Find top-K peaks
                top_k_values, top_k_indices = torch.topk(density_flat[b], min(self.num_clusters, density_flat.size(1)))
                centers = features_flat[b, top_k_indices]  # [K, C]
                densities = top_k_values  # [K]
                
                k = min(self.num_clusters, centers.size(0))
                self.temp_centers[:k, :].copy_(centers[:k])
                self.temp_densities[:k].copy_(densities[:k])
        
        return self.temp_centers, self.temp_densities
    
    def compute_semantic_density(self, features, cluster_centers, cluster_densities):
        """Compute semantic density map using soft assignment and prototype priors"""
        B, C, H, W = features.shape
        features_flat = features.view(B, C, -1).permute(0, 2, 1).contiguous()  # [B, HW, C]
        
        if cluster_centers.size(0) == 0:
            return self.zero_tensor.expand(B, 1, H, W)
        
        # Compute distances to cluster centers
        distances = torch.cdist(features_flat, cluster_centers)  # [B, HW, K]
        
        # Soft assignment with temperature scaling
        soft_assign = F.softmax(-distances / self.temperature, dim=-1)  # [B, HW, K]
        
        # Density-based priors
        priors = cluster_densities / (cluster_densities.sum() + 1e-8)  # [K]
        
        # Semantic strength
        semantic_strength = torch.sum(soft_assign * priors.unsqueeze(0).unsqueeze(0), dim=-1)  # [B, HW]
        
        return semantic_strength.view(B, 1, H, W)
    
    def forward(self, features):
        """
        Args:
            features: [B, C, H, W]
        Returns:
            semantic_density_map: [B, 1, H, W]
        """
        B, C, H, W = features.shape
    
        projected_features = self.feature_projector(features)
    
        if not self.training:
            return torch.zeros(B, 1, H, W, device=features.device, dtype=features.dtype)
    
        # Update memory bank
        self.update_memory_bank(projected_features)
    
        # Update clusters periodically
        if self.update_counter.item() % self.update_freq == 0:
            # Compute density using k-NN adaptive bandwidth
            density_map = self.compute_density_knn(projected_features, self.memory_bank)
        
            # Find density peaks
            centers, densities = self.find_density_peaks(projected_features, density_map)
        
            # Update cluster centers
            if centers.size(0) > 0:
                with torch.no_grad():
                    # Keep top-K clusters
                    if centers.size(0) >= self.num_clusters:
                        top_indices = torch.topk(densities, self.num_clusters).indices
                        self.cluster_centers[:self.num_clusters].copy_(centers[top_indices])
                        self.cluster_densities[:self.num_clusters].copy_(densities[top_indices])
                        self.num_active_clusters.fill_(self.num_clusters)
                    else:
                        self.cluster_centers[:centers.size(0)].copy_(centers)
                        self.cluster_densities[:centers.size(0)].copy_(densities)
                        self.num_active_clusters.fill_(centers.size(0))
    
        with torch.no_grad():
            self.update_counter.add_(1)
    
        # Compute semantic density map
        active_centers = self.cluster_centers[:self.num_active_clusters.item()]
        active_densities = self.cluster_densities[:self.num_active_clusters.item()]
    
        semantic_density_map = self.compute_semantic_density(
            projected_features, active_centers, active_densities
        )
    
        return semantic_density_map



class SS2D_DSM(nn.Module):
    """
    SS2D with Density-aware Sequence Modulation (DSM) support
    """
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            dt_rank="auto",
            act_layer=nn.SiLU,
            # dwconv ===============
            d_conv=3,  # < 2 means no conv
            conv_bias=True,
            # ======================
            dropout=0.0,
            bias=False,
            # ======================
            forward_type="v2",
            # DSM parameters
            enable_dsm=False,
            dsm_modulation_scale=0.1,
            **kwargs,
    ):
        """
        ssm_rank_ratio would be used in the future...
        """
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        d_expand = int(ssm_ratio * d_model)
        d_inner = int(min(ssm_rank_ratio, ssm_ratio) * d_model) if ssm_rank_ratio > 0 else d_expand
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.K = 4
        self.enable_dsm = enable_dsm

        # tags for forward_type ==============================
        def checkpostfix(tag, value):
            ret = value[-len(tag):] == tag
            if ret:
                value = value[:-len(tag)]
            return ret, value

        self.disable_force32, forward_type = checkpostfix("no32", forward_type)
        self.disable_z, forward_type = checkpostfix("noz", forward_type)
        self.disable_z_act, forward_type = checkpostfix("nozact", forward_type)

        self.out_norm = nn.LayerNorm(d_inner)

        # forward_type debug =======================================
        FORWARD_TYPES = dict(
            v2=partial(self.forward_corev2, force_fp32=None, SelectiveScan=SelectiveScanCore),
            v2_dsm=partial(self.forward_corev2_dsm, force_fp32=None, SelectiveScan=SelectiveScanCore),
        )
        self.forward_core = FORWARD_TYPES.get(forward_type, FORWARD_TYPES.get("v2", None))

        # in proj =======================================
        d_proj = d_expand if self.disable_z else (d_expand * 2)
        self.in_proj = nn.Conv2d(d_model, d_proj, kernel_size=1, stride=1, groups=1, bias=bias, **factory_kwargs)
        self.act: nn.Module = nn.GELU()

        # conv =======================================
        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=d_expand,
                out_channels=d_expand,
                groups=d_expand,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )

        # rank ratio =====================================
        self.ssm_low_rank = False
        if d_inner < d_expand:
            self.ssm_low_rank = True
            self.in_rank = nn.Conv2d(d_expand, d_inner, kernel_size=1, bias=False, **factory_kwargs)
            self.out_rank = nn.Linear(d_inner, d_expand, bias=False, **factory_kwargs)

        # x proj ============================
        self.x_proj = [
            nn.Linear(d_inner, (self.dt_rank + self.d_state * 2), bias=False,
                      **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        # out proj =======================================
        self.out_proj = nn.Conv2d(d_expand, d_model, kernel_size=1, stride=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

        # simple init dt_projs, A_logs, Ds
        self.Ds = nn.Parameter(torch.ones((self.K * d_inner)))
        self.A_logs = nn.Parameter(
            torch.zeros((self.K * d_inner, self.d_state)))  # A == -A_logs.exp() < 0; # 0 < exp(A * dt) < 1
        self.dt_projs_weight = nn.Parameter(torch.randn((self.K, d_inner, self.dt_rank)))
        self.dt_projs_bias = nn.Parameter(torch.randn((self.K, d_inner)))

        # DSM module
        if self.enable_dsm:
            self.dsm = DensityAwareModulation(modulation_scale=dsm_modulation_scale)

    def forward_corev2(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None):
        """
        Standard forward core without DSM
        """
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
        x = cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds,
            out_norm=getattr(self, "out_norm", None),
            out_norm_shape=getattr(self, "out_norm_shape", "v0"),
            delta_softplus=True, force_fp32=force_fp32,
            SelectiveScan=SelectiveScan, ssoflex=self.training,  # output fp32
        )
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward_corev2_dsm(self, x: torch.Tensor, channel_first=False, SelectiveScan=SelectiveScanCore,
                       cross_selective_scan=cross_selective_scan, force_fp32=None, semantic_density_map=None):
        """
        Forward core with DSM support - modulates SSM parameters (delta and B)
        """
        force_fp32 = (self.training and (not self.disable_force32)) if force_fp32 is None else force_fp32
        if not channel_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        if self.ssm_low_rank:
            x = self.in_rank(x)
    
        # Apply DSM modulation if enabled
        if self.enable_dsm and semantic_density_map is not None:
            x = cross_selective_scan_with_dsm(
                x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
                self.A_logs, self.Ds,
                out_norm=getattr(self, "out_norm", None),
                out_norm_shape=getattr(self, "out_norm_shape", "v0"),
                delta_softplus=True, force_fp32=force_fp32,
                SelectiveScan=SelectiveScan, ssoflex=self.training,
                semantic_density_map=semantic_density_map,
                dsm=self.dsm
            )
        else:
            x = cross_selective_scan(
                x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
                self.A_logs, self.Ds,
                out_norm=getattr(self, "out_norm", None),
                out_norm_shape=getattr(self, "out_norm_shape", "v0"),
                delta_softplus=True, force_fp32=force_fp32,
                SelectiveScan=SelectiveScan, ssoflex=self.training,
            )
    
        if self.ssm_low_rank:
            x = self.out_rank(x)
        return x

    def forward(self, x: torch.Tensor, semantic_density_map=None, **kwargs):
        x = self.in_proj(x)
        if not self.disable_z:
            x, z = x.chunk(2, dim=1)  # (b, d, h, w)
            if not self.disable_z_act:
                z1 = self.act(z)
        if self.d_conv > 0:
            x = self.conv2d(x)  # (b, d, h, w)
        x = self.act(x)
        
        # Use DSM-enabled forward core if DSM is enabled
        if self.enable_dsm:
            y = self.forward_corev2_dsm(x, channel_first=(self.d_conv > 1), semantic_density_map=semantic_density_map)
        else:
            y = self.forward_core(x, channel_first=(self.d_conv > 1))
        
        y = y.permute(0, 3, 1, 2).contiguous()
        if not self.disable_z:
            y = y * z1
        out = self.dropout(self.out_proj(y))
        return out



class VSSBlock_DSM(nn.Module):
    """
    VSSBlock with Density-aware Sequence Modulation (DSM)
    Integrates semantic density map from ODPC to dynamically adjust SSM parameters
    """
    def __init__(
            self,
            in_channels: int = 0,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
            # =============================
            ssm_d_state: int = 16,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank: Any = "auto",
            ssm_act_layer=nn.SiLU,
            ssm_conv: int = 3,
            ssm_conv_bias=True,
            ssm_drop_rate: float = 0,
            ssm_init="v0",
            forward_type="v2",
            # =============================
            mlp_ratio=4.0,
            mlp_act_layer=nn.GELU,
            mlp_drop_rate: float = 0.0,
            # =============================
            use_checkpoint: bool = False,
            post_norm: bool = False,
            # DSM parameters
            enable_dsm: bool = True,
            dsm_modulation_scale: float = 0.1,
            # ODPC parameters for semantic density map
            enable_odpc: bool = True,
            odpc_num_clusters: int = 8,
            odpc_k_neighbors: int = 20,
            odpc_alpha: float = 1.0,
            odpc_memory_size: int = 1000,
            odpc_update_freq: int = 10,
            **kwargs,
    ):
        super().__init__()
        self.ssm_branch = ssm_ratio > 0
        self.mlp_branch = mlp_ratio > 0
        self.use_checkpoint = use_checkpoint
        self.post_norm = post_norm
        self.enable_dsm = enable_dsm
        self.enable_odpc = enable_odpc

        # proj
        self.proj_conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU()
        )

        if self.ssm_branch:
            self.norm = norm_layer(hidden_dim)
            self.op = SS2D_DSM(
                d_model=hidden_dim,
                d_state=ssm_d_state,
                ssm_ratio=ssm_ratio,
                ssm_rank_ratio=ssm_rank_ratio,
                dt_rank=ssm_dt_rank,
                act_layer=ssm_act_layer,
                d_conv=ssm_conv,
                conv_bias=ssm_conv_bias,
                dropout=ssm_drop_rate,
                initialize=ssm_init,
                forward_type=forward_type,
                enable_dsm=enable_dsm,
                dsm_modulation_scale=dsm_modulation_scale,
            )

        self.drop_path = DropPath(drop_path)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)
        
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=mlp_act_layer,
                               drop=mlp_drop_rate, channels_first=False)

        # ODPC module for semantic density map
        if self.enable_odpc:
            self.odpc = OnlineDPCClus(
                feature_dim=hidden_dim,
                num_clusters=odpc_num_clusters,
                k_neighbors=odpc_k_neighbors,
                alpha=odpc_alpha,
                memory_size=odpc_memory_size,
                update_freq=odpc_update_freq
            )
        else:
            self.odpc = None

    def forward(self, input: torch.Tensor):
        input = self.proj_conv(input)
        X1 = self.lsblock(input)
        
        if self.ssm_branch:
            # Get semantic density map from ODPC
            semantic_density_map = None
            if self.enable_odpc and self.odpc is not None:
                semantic_density_map = self.odpc(input)
            
            # Apply SSM with optional DSM modulation
            x = input + self.drop_path(self.op(self.norm(X1), semantic_density_map=semantic_density_map))
        else:
            x = input
        
        if self.mlp_branch:
            x = x + self.drop_path(self.mlp(self.norm2(x)))  # FFN
        return x



class SEIN(nn.Module):
    """
    SE-IN (Spatial Enhancement with Instance Normalization)
    Integrates semantic prior from ODPC for density weighted saliency recalibration
    """
    def __init__(self, 
                 c1: int,  # in_channels
                 c2: int,  # out_channels  
                 reduction: int = 16,
                 kernel_size: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 groups: int = None,
                 instance_norm_eps: float = 1e-5,
                 instance_norm_affine: bool = True,
                 gaussian_sigma: float = 1.0):
        super().__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.reduction = reduction
        self.gaussian_sigma = gaussian_sigma
        
        if groups is None:
            groups = c2
        
        # Instance Normalization
        self.instance_norm = nn.InstanceNorm2d(
            c2, eps=instance_norm_eps, affine=instance_norm_affine
        )
        
        # Depthwise separable convolution for structure enhancement
        self.depthwise_conv = nn.Conv2d(
            c2, c2, kernel_size=kernel_size, stride=stride, 
            padding=padding, groups=groups, bias=False
        )
        self.pointwise_conv = nn.Conv2d(c2, c2, 1, bias=False)
        
        # Spatial mask generation (3-layer conv + Gaussian smoothing)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1, bias=False),
            nn.Sigmoid()
        )
        
        # SE channel attention
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        hidden_dim = max(1, c2 // reduction)
        self.se_fc = nn.Sequential(
            nn.Linear(c2, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, c2, bias=False)
        )
        
        self.final_conv = nn.Conv2d(c2 * 2, c2, 1, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.activation = nn.SiLU()
        
        # Input projection if needed
        if c1 != c2:
            self.in_proj = nn.Sequential(
                nn.Conv2d(c1, c2, 1, bias=False),
                nn.BatchNorm2d(c2),
                nn.ReLU(inplace=True)
            )
        else:
            self.in_proj = nn.Identity()
    
    def gaussian_smooth(self, x, sigma):
        """Apply 2D Gaussian smoothing"""
        kernel_size = int(2 * sigma * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        coords = torch.arange(kernel_size, dtype=torch.float32, device=x.device)
        coords = coords - kernel_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        
        # Create 2D kernel
        kernel = g[:, None] * g[None, :]
        kernel = kernel.view(1, 1, kernel_size, kernel_size)
        kernel = kernel.to(dtype=x.dtype)
        
        # Apply convolution
        padding = kernel_size // 2
        return F.conv2d(x, kernel, padding=padding)
    
    def forward(self, x, semantic_density_map=None):
        """
        Args:
            x: [B, C, H, W] input features
            semantic_density_map: [B, 1, H, W] from ODPC
        Returns:
            output: [B, C, H, W] enhanced features
        """
        x = self.in_proj(x)
        
        x_norm = self.instance_norm(x)
        
        z = self.pointwise_conv(self.depthwise_conv(x_norm))
        
        if semantic_density_map is not None:
            if semantic_density_map.shape[-2:] != x.shape[-2:]:
                semantic_density_map = F.interpolate(
                    semantic_density_map, size=x.shape[-2:], 
                    mode='bilinear', align_corners=False
                )
            
            spatial_mask = self.spatial_conv(semantic_density_map)
  
            spatial_mask = self.gaussian_smooth(spatial_mask, self.gaussian_sigma)
        else:
            spatial_mask = torch.ones_like(x[:, :1])
        
        # SE channel attention
        se_weights = self.se_fc(self.global_avg_pool(z).view(z.size(0), -1))
        se_weights = torch.sigmoid(se_weights).view(z.size(0), -1, 1, 1)
        
        # spatial mask and channel attention
        z_enhanced = z * spatial_mask * se_weights
        
        # Residual fusion with concatenation
        concat_features = torch.cat([x_norm, x_norm + z_enhanced], dim=1)
        output = self.final_conv(concat_features)
        output = self.bn(output)
        output = self.activation(output)
        
        return output


class SEINBlock(nn.Module):

    def __init__(self, 
                 c1: int,  # in_channels
                 c2: int,  # out_channels
                 drop_path: float = 0.0,
                 norm_layer: Callable[..., torch.nn.Module] = partial(LayerNorm2d, eps=1e-6),
                 # ODPC parameters
                 enable_odpc: bool = True,
                 odpc_num_clusters: int = 8,
                 odpc_k_neighbors: int = 20,  
                 odpc_alpha: float = 1.0,     
                 odpc_memory_size: int = 1000,
                 odpc_update_freq: int = 10,
                 # SEIN parameters
                 sein_reduction: int = 16,
                 sein_kernel_size: int = 3,
                 sein_stride: int = 1,
                 sein_padding: int = 1,
                 sein_groups: int = None,
                 sein_instance_norm_eps: float = 1e-5,
                 sein_instance_norm_affine: bool = True,
                 sein_gaussian_sigma: float = 1.0,
                 **kwargs):
        super().__init__()
        
        self.c1 = c1
        self.c2 = c2
        self.enable_odpc = enable_odpc
        
        self.in_proj = nn.Sequential(
            nn.Conv2d(c1, c2, 1, bias=False),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True)
        ) if c1 != c2 else nn.Identity()
        
        self.norm = norm_layer(c2)
        
        # ODPC module
        if self.enable_odpc:
            self.odpc = OnlineDPCClus(
                feature_dim=c2,
                num_clusters=odpc_num_clusters,
                k_neighbors=odpc_k_neighbors,  
                alpha=odpc_alpha,              
                memory_size=odpc_memory_size,
                update_freq=odpc_update_freq
            )
        else:
            self.odpc = None
        
        # SEIN module
        self.sein = SEIN(
            c1=c2,  # Use c2 as both input and output
            c2=c2,
            reduction=sein_reduction,
            kernel_size=sein_kernel_size,
            stride=sein_stride,
            padding=sein_padding,
            groups=sein_groups,
            instance_norm_eps=sein_instance_norm_eps,
            instance_norm_affine=sein_instance_norm_affine,
            gaussian_sigma=sein_gaussian_sigma
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] input features
        Returns:
            output: [B, C, H, W] enhanced features
        """
        x = self.in_proj(x)
        
        # Get semantic density map from ODPC
        semantic_density_map = None
        if self.enable_odpc and self.odpc is not None:
            semantic_density_map = self.odpc(x)
        
        x_norm = self.norm(x)
        
        enhanced = self.sein(x_norm, semantic_density_map)
        
        residual = enhanced - x_norm
        output = x + self.drop_path(residual)
        
        return output