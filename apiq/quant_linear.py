import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CLIPMIN = 1e-5

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        metric = "minmax",
        group_size = None,
        shape = None,
        lwc = False,
        disable_zero_point = False,
    ):
        super().__init__()
        assert 1 <= n_bits <= 16, "bitwidth not supported"
        assert shape[-1] % group_size == 0, "group size should be divisible by the in_feature"
        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        self.n_bits = n_bits
        self.group_size = group_size
        self.metric = metric
        self.lwc = lwc
        
        if self.disable_zero_point:
            self.qmin = - (2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** (n_bits) - 1
        
        self.scale = None
        self.zero_point = None
        self.round_zero_point = None
        
        init_value = 4.  # init value of learnable weight clipping
        if lwc:
            dim1 = int(shape[0] * math.ceil(shape[1] / group_size)) if group_size else shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
        
    def forward(self, x: torch.Tensor):
        if self.n_bits >= 16:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2 ** self.n_bits-1).round_().div_(2 ** self.n_bits-1)
        self.calibration(x)
        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant

    def fake_quant(self, x, scale, round_zero_point):
        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)
        #scale_zeros = round_zero_point * scale
        #x_int = round_ste((x + scale_zeros) / scale)
        x_int = round_ste(x / scale)
        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int
        x_dequant = x_dequant.sub(round_zero_point)
        x_dequant = x_dequant.mul(scale)
        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        return x_dequant

    def calibration(self, x):
        if self.group_size:
            x = x.reshape(-1, self.group_size)
        reduce_shape = [-1]
        xmin = x.amin(reduce_shape, keepdim=True)
        xmax =  x.amax(reduce_shape, keepdim=True)
        if self.lwc:
            xmax = F.sigmoid(self.upbound_factor) * xmax
            xmin = F.sigmoid(self.lowbound_factor) * xmin
            
        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            range = xmax - xmin
            scale = range / (2 ** self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = - xmin / self.scale
        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()
        
    def register_scales_and_zeros(self):
        self.register_buffer('scales', self.scale)
        self.register_buffer('zeros', self.round_zero_point)
        del self.scale
        del self.round_zero_point


class QuantLinear(nn.Module):
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
    ):
        super().__init__()
        self.register_buffer('weight', org_module.weight)
        if org_module.bias is not None:
            self.register_buffer('bias', org_module.bias)
        else:
            self.bias = None
        self.in_features = org_module.in_features
        self.out_features = org_module.out_features
        self.use_weight_quant = False
        self.use_temporary_parameter = False
        self.weight_quantizer = UniformAffineQuantizer(**weight_quant_params, shape=org_module.weight.shape)

    def forward(self, x: torch.Tensor):
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias
        elif self.use_weight_quant:
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            weight = self.weight
            bias = self.bias
        out = F.linear(x, weight, bias)
        return out

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant
