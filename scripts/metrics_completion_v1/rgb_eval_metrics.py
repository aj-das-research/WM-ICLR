"""Frozen pixel/perceptual conventions; every result is per RGB frame."""
from pathlib import Path
import sys
import torch
from torch.nn import functional as F

METRICS=('rgb_mse','psnr_db','ssim','lpips_vgg','uiqi')
DIRECTION={'rgb_mse':'lower','psnr_db':'higher','ssim':'higher','lpips_vgg':'lower','uiqi':'higher'}
CONVENTIONS={
 'range':'prediction clamp[0,1]; target resized uint8 /255; 224x224 all RGB pixels, no mask',
 'rgb_mse':'float64 mean squared difference over3x224x224; units[0,1]^2',
 'psnr_db':'-10log10(per-frame RGB MSE), data_range1; exact zero MSE gives +inf, never capped',
 'ssim':'Wang SSIM; 11x11 separable Gaussian sigma1.5; valid windows; population weighted variance; K1=.01,K2=.03,L=1; mean across windows and RGB channels; float64',
 'uiqi':'Wang-Bovik universal Q;8x8 uniform valid windows; population variance/covariance; mean windows/RGB channels; zero variance<=1e-12 gives contrast1, zero squared-means<=1e-12 gives luminance1; float64; signed values retained',
 'lpips_vgg':'official lpips0.1.4 learned calibrationv0.1, ImageNet VGG16; normalize=True maps[0,1]to[-1,1]; FP32 eval/frozen; spatial=False; no resizing inside LPIPS',
 'relative_gains':'only RGBMSE and LPIPS (lower better); PSNR/SSIM/UIQI use signed absolute differences in their own units',
}


def require(ok,message):
    if not ok:raise ValueError(message)


def check_images(a,b):
    require(a.shape==b.shape and a.ndim==4 and a.shape[1:]==(3,224,224),'Expected matched Nx3x224x224 RGB')
    require(a.dtype==b.dtype==torch.float32 and torch.isfinite(a).all() and torch.isfinite(b).all(),'RGB must be finite FP32')
    require((a>=0).all() and (a<=1).all() and (b>=0).all() and (b<=1).all(),'RGB outside[0,1]')


def moments(a,b,kernel):
    filt=lambda x:F.conv2d(x,kernel.expand(3,1,*kernel.shape[-2:]),groups=3)
    ma,mb=filt(a),filt(b)
    va=(filt(a*a)-ma*ma).clamp_min(0)
    vb=(filt(b*b)-mb*mb).clamp_min(0)
    cov=filt(a*b)-ma*mb
    return ma,mb,va,vb,cov


def pixel_metrics(prediction,target):
    check_images(prediction,target)
    a,b=prediction.double(),target.double()
    mse=(a-b).square().mean((1,2,3))
    psnr=-10*torch.log10(mse)
    coords=torch.arange(11,device=a.device,dtype=torch.float64)-5
    gaussian=torch.exp(-coords.square()/(2*1.5**2));gaussian/=gaussian.sum()
    kernel=(gaussian[:,None]*gaussian[None,:])[None,None]
    ma,mb,va,vb,cov=moments(a,b,kernel)
    ssim=((2*ma*mb+.01**2)*(2*cov+.03**2)/((ma*ma+mb*mb+.01**2)*(va+vb+.03**2))).mean((1,2,3))
    uniform=torch.ones((1,1,8,8),device=a.device,dtype=torch.float64)/64
    ma,mb,va,vb,cov=moments(a,b,uniform)
    variance=va+vb;means=ma.square()+mb.square()
    contrast=torch.where(variance<=1e-12,torch.ones_like(variance),2*cov/variance.clamp_min(1e-30))
    luminance=torch.where(means<=1e-12,torch.ones_like(means),2*ma*mb/means.clamp_min(1e-30))
    uiqi=(contrast*luminance).mean((1,2,3))
    return {'rgb_mse':mse,'psnr_db':psnr,'ssim':ssim,'uiqi':uiqi}


def load_lpips(base,device):
    """No download and no global torchvision monkeypatch; fill official slices."""
    base=Path(base);sys.path.insert(0,str(base/'vendor'))
    import lpips
    require(Path(lpips.__file__).resolve().is_relative_to((base/'vendor').resolve()),'LPIPS imported from unregistered location')
    metric=lpips.LPIPS(net='vgg',version='0.1',pnet_rand=True,
        model_path=str(base/'vendor/lpips/weights/v0.1/vgg.pth'),verbose=False)
    # pnet_rand suppresses implicit network access; replace EVERY trunk parameter.
    pretrained=torch.load(base/'assets/vgg16-397923af.pth',map_location='cpu',weights_only=True)
    current=metric.net.state_dict();replacement={}
    for key,value in current.items():
        _,layer,param=key.split('.')
        source=f'features.{layer}.{param}'
        require(source in pretrained and pretrained[source].shape==value.shape,'Official VGG slice mapping mismatch')
        replacement[key]=pretrained[source]
    metric.net.load_state_dict(replacement,strict=True)
    require(all(torch.equal(v,pretrained['features.'+'.'.join(k.split('.')[1:])]) for k,v in metric.net.state_dict().items()),'VGG trunk incomplete')
    metric.pnet_rand=False  # all random trunk values have now been replaced.
    return metric.to(device).eval().requires_grad_(False)


@torch.inference_mode()
def frame_metrics(prediction,target,perceptual):
    values=pixel_metrics(prediction,target)
    values['lpips_vgg']=perceptual(prediction,target,normalize=True).flatten().double()
    require(values['lpips_vgg'].shape==(len(prediction),) and torch.isfinite(values['lpips_vgg']).all(),'Invalid LPIPS outputs')
    return torch.stack([values[m] for m in METRICS],dim=-1)
