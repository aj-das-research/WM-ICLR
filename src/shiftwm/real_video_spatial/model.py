"""Causal patch-grid controls using pinned LeWM temporal attention.

Coordinates are channel-major 4x4 DINO patches. Spatial transport operates in
shared per-channel training-normalized coordinates, never position-specific
normalizations. Outputs are latent features, not RGB or physical robot states.
"""
from dataclasses import asdict, dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from shiftwm.model import ResidualFiLM, TransitionContext
from shiftwm.upstream import _source_module


@dataclass
class SpatialConfig:
    feature_dim: int = 6144
    channels: int = 384
    grid_size: int = 4
    action_dim: int = 35
    hidden_dim: int = 96
    history_length: int = 3
    depth: int = 4
    context_dim: int = 32
    context_hidden: int = 128
    innovation_bound: float = 1.0
    identity_bias: float = 4.0
    initial_gate_logit: float = -3.0
    mode: str = "transport"


class SpatialWorldModel(nn.Module):
    MODES = ("autoregressive", "anchored_additive", "transport", "context_off", "action_free")

    def __init__(self, config, feature_mean, feature_std, action_mean, action_std):
        super().__init__()
        self.config = SpatialConfig(**config) if isinstance(config, dict) else config
        c=self.config; n=c.grid_size**2; d=c.hidden_dim
        if (c.mode not in self.MODES or c.feature_dim != c.channels*n or c.history_length != 3
                or c.grid_size != 4 or d % 6 or not 0 < c.innovation_bound <= 10
                or not math.isfinite(c.identity_bias) or not math.isfinite(c.initial_gate_logit)):
            raise ValueError("Invalid spatial model configuration")
        for key, value, dim in (("feature_mean",feature_mean,c.feature_dim),("feature_std",feature_std,c.feature_dim),
                                ("action_mean",action_mean,c.action_dim),("action_std",action_std,c.action_dim)):
            value=torch.as_tensor(value,dtype=torch.float32).reshape(-1)
            if len(value)!=dim or not torch.isfinite(value).all() or (key.endswith("std") and (value<=0).any()):
                raise ValueError("Invalid normalization: "+key)
            if key.startswith("feature") and not torch.equal(value.view(c.channels,n),value.view(c.channels,n)[:,:1].expand(-1,n)):
                raise ValueError("Spatial transport requires shared per-channel normalization")
            self.register_buffer(key,value)
        self.input_projection=nn.Sequential(nn.LayerNorm(c.channels),nn.Linear(c.channels,d))
        self.position=nn.Parameter(torch.zeros(1,n,d)); nn.init.normal_(self.position,std=.01)
        self.spatial_norm=nn.LayerNorm(d)
        self.spatial_attention=nn.MultiheadAttention(d,6,dropout=0.0,batch_first=True)
        self.spatial_ff=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,2*d),nn.GELU(),nn.Linear(2*d,d))
        self.action_prefix=nn.GRU(c.action_dim,d,batch_first=True)
        modules=_source_module("shiftwm_upstream_module","module.py")
        self.predictor=modules.ARPredictor(num_frames=3,input_dim=d,hidden_dim=d,output_dim=d,
            depth=c.depth,heads=6,mlp_dim=4*d,dim_head=16,dropout=.1,emb_dropout=0.)
        self.context=TransitionContext(d,c.action_dim,c.context_hidden,c.context_dim)
        self.context_adapter=ResidualFiLM(d,c.context_dim)
        self.output_projection=nn.Sequential(nn.LayerNorm(d),nn.Linear(d,c.channels))
        nn.init.zeros_(self.output_projection[-1].weight); nn.init.zeros_(self.output_projection[-1].bias)
        # Heads are instantiated for all arms, but inactive heads are excluded
        # from trainable counts/optimizer. Do not claim exact parameter equality.
        self.transport_query=nn.Linear(d,d,bias=False)
        self.transport_key=nn.Linear(d,d,bias=False)
        self.gate=nn.Linear(d,1)
        nn.init.zeros_(self.gate.weight); nn.init.constant_(self.gate.bias,c.initial_gate_logit)
        self.register_buffer("identity_transport",torch.eye(n))
        transport=c.mode in ("transport","context_off","action_free")
        for module in (self.transport_query,self.transport_key,self.gate): module.requires_grad_(transport)
        if c.mode=="context_off":
            self.context.requires_grad_(False); self.context_adapter.requires_grad_(False)
        if c.mode=="action_free": self.action_prefix.requires_grad_(False)

    def tokens(self, flat):
        c=self.config
        if flat.shape[-1]!=c.feature_dim: raise ValueError("Wrong channel-major feature dimension")
        return flat.reshape(*flat.shape[:-1],c.channels,c.grid_size**2).transpose(-1,-2)

    def flatten(self,tokens):
        return tokens.transpose(-1,-2).reshape(*tokens.shape[:-2],self.config.feature_dim)

    def normalize_features(self,x): return (x.float()-self.feature_mean)/self.feature_std

    def normalize_actions(self,x):
        return torch.zeros_like(x,dtype=torch.float32) if self.config.mode=="action_free" else (x.float()-self.action_mean)/self.action_std

    def _encode(self,x):
        z=self.input_projection(x)+self.position
        shape=z.shape
        z=z.reshape(-1,shape[-2],shape[-1]); v=self.spatial_norm(z)
        z=z+self.spatial_attention(v,v,v,need_weights=False)[0]
        return (z+self.spatial_ff(z)).reshape(shape)

    def _predict_normalized(self,support,past_actions,future_actions,return_details=False):
        c=self.config; b=support.shape[0]; n=c.grid_size**2
        if (support.ndim!=3 or support.shape[1:]!=(3,c.feature_dim)
                or past_actions.shape!=(b,2,c.action_dim) or future_actions.ndim!=3
                or future_actions.shape[0]!=b or future_actions.shape[1]<1 or future_actions.shape[2]!=c.action_dim):
            raise ValueError("Expected three observed frames, two past actions, and a nonempty future action prefix")
        past=self.normalize_actions(past_actions); future=self.normalize_actions(future_actions)
        # Unidirectional GRU: prefix outputs never depend on subsequent actions.
        action_states=self.action_prefix(torch.cat((past,future),1))[0]
        if c.mode=="action_free": action_states=torch.zeros_like(action_states)
        observed=self.tokens(support); anchor=observed[:,-1]
        encoded_support=self._encode(observed)
        context=None if c.mode=="context_off" else self.context(encoded_support.mean(2),past)
        anchor_encoded=encoded_support[:,-1]
        observations=observed; predictions=[]; details=[]
        for t in range(future.shape[1]):
            encoded=self._encode(observations) if c.mode=="autoregressive" else encoded_support
            if context is not None:
                encoded=self.context_adapter(encoded.reshape(b,3*n,c.hidden_dim),context).reshape(b,3,n,c.hidden_dim)
            action_history=action_states[:,t:t+3] if c.mode=="autoregressive" else torch.cat((action_states[:,:2],action_states[:,t+2:t+3]),1)
            temporal=encoded.transpose(1,2).reshape(b*n,3,c.hidden_dim)
            actions=action_history[:,None].expand(b,n,3,c.hidden_dim).reshape(b*n,3,c.hidden_dim)
            hidden=self.predictor(temporal,actions)[:,-1].reshape(b,n,c.hidden_dim)
            residual=self.output_projection(hidden).float()
            if c.mode=="autoregressive":
                value=observations[:,-1]+residual
                observations=torch.cat((observations[:,1:],value[:,None]),1)
            elif c.mode=="anchored_additive": value=anchor+residual
            else:
                scores=self.transport_query(hidden)@self.transport_key(anchor_encoded).transpose(-1,-2)/math.sqrt(c.hidden_dim)
                scores=scores.float()+c.identity_bias*self.identity_transport
                transport=torch.softmax(scores,-1)
                gate=torch.sigmoid(self.gate(hidden).float())
                innovation=c.innovation_bound*torch.tanh(residual)
                value=(1-gate)*anchor+gate*(transport@anchor)+innovation
                if return_details: details.append({"transport":transport,"gate":gate,"innovation":innovation})
            predictions.append(value)
        result=self.flatten(torch.stack(predictions,1))
        return (result,details) if return_details else result

    def predict(self,support_features,support_actions,future_actions):
        normalized=self._predict_normalized(self.normalize_features(support_features),support_actions,future_actions)
        return normalized*self.feature_std+self.feature_mean

    def forward(self,batch):
        x=batch["features"]; actions=batch["actions"]
        if x.ndim!=3 or x.shape[1]<4 or actions.shape[:2]!=(len(x),x.shape[1]-1): raise ValueError("Invalid chronological batch")
        target=self.normalize_features(x[:,3:])
        prediction=self._predict_normalized(self.normalize_features(x[:,:3]),actions[:,:2],actions[:,2:])
        return {"loss":F.mse_loss(prediction,target),"standardized_predictions":prediction,"standardized_targets":target,
                "predictions":prediction*self.feature_std+self.feature_mean,"targets":x[:,3:]}

    @property
    def package_config(self):
        return {"format_version":1,"model_config":asdict(self.config),"coordinate_layout":"channel_major_384x4x4_shared_channel_normalization",
                **{k:getattr(self,k).detach().cpu().tolist() for k in ("feature_mean","feature_std","action_mean","action_std")}}


def from_config(config):
    if config.get("format_version")!=1 or config.get("coordinate_layout")!="channel_major_384x4x4_shared_channel_normalization": raise ValueError("Unsupported spatial package")
    return SpatialWorldModel(config["model_config"],**{k:config[k] for k in ("feature_mean","feature_std","action_mean","action_std")})


def pool_to_original_2x2(features):
    if features.shape[-1]!=6144: raise ValueError("Expected channel-major 4x4 features")
    shape=features.shape[:-1]
    return F.avg_pool2d(features.reshape(-1,384,4,4),2).reshape(*shape,1536)
