import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn import functional as F
import torch.distributed as dist


# Uniform quantization
def uniform_symmetric_quantizer_per_tensor(x, bits=8, scale_bits=8, act_max=None):
    # per_tensor, signed=True
    # calculate scale
    if act_max == None:
        maxv = torch.max(torch.abs(x))
    else:
        maxv = act_max

    minv = - maxv
    num_levels = 2 ** bits
    scale = (maxv - minv) / (num_levels - 2)
    scale_int = get_scale_approximation(scale, scale_bits)
    # print('scale_int={}'.format(scale_int))

    # clamp
    x = torch.clamp(x, min=minv.item(), max=maxv.item())
    # quantize
    x_int = RoundFunction.apply(x / scale_int)
    # dequantize
    x_dequant = x_int * scale_int

    return x_dequant, scale_int

def uniform_symmetric_quantizer_per_channel(x, bits=8, scale_bits=8, scale_int_bias=None):
    # per_channel, signed=True
    # calculate scale
    if scale_int_bias is not None:
        # quantize the bias
        scale_int = scale_int_bias
        # quantize
        x_int = RoundFunction.apply(x / scale_int)
        # # clamp
        # x_int = torch.clamp(x_int, min= - 2 ** (bits-1) + 1, max= 2 ** (bits-1) - 1)
        # dequantize
        x_dequant = x_int * scale_int

        return x_dequant

    else:
        c_out, c_in, k, w = x.shape
        x_reshape = x.reshape(c_out, -1)
        maxv, _ = torch.max(torch.abs(x_reshape), 1)   # maxv shape = c_out*1
        # print('maxv.shape={},\nmaxv={}'.format(maxv.shape, maxv))
        minv = - maxv
        num_levels = 2 ** bits
        scale = (maxv - minv) / (num_levels - 2)   # scale shape = c_out*1
        # print('scale.shape={},\nscale={}'.format(scale.shape, scale))
        scale_int = get_scale_approximation(scale, scale_bits)
        # print('scale_int.shape={},\nscale_int={}'.format(scale_int.shape, scale_int))
        scale_int_expand = scale_int.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)

        # clamp
        # for i in range(c_out):
        #     x[i] = torch.clamp(x[i], min=float(minv[i]), max=float(maxv[i]))
        maxv = maxv.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
        minv = minv.unsqueeze(1).unsqueeze(2).unsqueeze(3).expand_as(x)
        x = torch.where(x > minv, x, minv)
        x = torch.where(x < maxv, x, maxv)

        # quantize
        x_int = RoundFunction.apply(x / scale_int_expand)
        # print('x_int={}'.format(x_int))
        # dequantize
        x_dequant = x_int * scale_int_expand

        return x_dequant, scale_int


class RoundFunction(Function):
    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def get_scale_approximation_shift_bits(fp32_scale, mult_bits):

    shift_bits = torch.floor(torch.log2((2 ** mult_bits - 1) / fp32_scale))
    # print('shift_bits.shape={},\nshift_bits={}'.format(shift_bits.shape, shift_bits))
    shift_bits = torch.min(mult_bits, shift_bits)
    # print('shift_bits.shape={},\nshift_bits={}'.format(shift_bits.shape, shift_bits))
    return shift_bits


def get_scale_approximation_mult(fp32_scale, shift_bits):
    return torch.floor(fp32_scale * (2 ** shift_bits))


def get_scale_approximation(fp32_scale, mult_bits):
    shift_bits = get_scale_approximation_shift_bits(fp32_scale, mult_bits)
    # print('shift_bits={}'.format(shift_bits))
    multiplier = get_scale_approximation_mult(fp32_scale, shift_bits)
    # print('multiplier={}'.format(multiplier))
    scale_int = multiplier / (2 ** shift_bits)
    # print('scale_int={}'.format(scale_int))
    return scale_int


class AllReduce(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, async_op=False)
        return grad_output


class Conv2d_quantization(nn.Conv2d):
    """
    custom convolutional layers for quantization
    """
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 quantization_bits=8, scale_bits=8, bias_bits=16, 
                 running_stat=True, beta=0.9, first_layer=False, stats_mode="", last_layer=False):
        super(Conv2d_quantization, self).__init__(in_channels, out_channels, kernel_size,
                                               stride, padding, dilation, groups, bias)

        # Sync setting
        assert stats_mode in [""]
        self._stats_mode = stats_mode

        self.register_buffer('quantization_bits', torch.Tensor([quantization_bits]))
        self.register_buffer('scale_bits', torch.Tensor([scale_bits]))
        self.register_buffer('bias_bits', torch.Tensor([bias_bits]))

        # EMA for activation
        self.running_stat = running_stat
        self.register_buffer('act_max', torch.zeros(1))
        self.register_buffer('beta', torch.Tensor([beta]))
        self.register_buffer('beta_t', torch.ones(1))

        self.first_layer = first_layer
        self.last_layer = last_layer

        # first layer and last layer quantized to 8 bit if quantization_bits <= 4
        if self.first_layer or self.last_layer and self.quantization_bits <= 4:
            self.quantization_bits = torch.Tensor([8])
            self.scale_bits = torch.Tensor([32])


    def fix_activation_range(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = False

    def unfix_activation_range(self):
        """
        fix the activation range by setting running stat
        """
        self.running_stat = True

    def forward(self, input):

        if self.first_layer:
            quantized_input = input

        else:
            # EMA for activation
            if self.running_stat:

                act_abs_max = input.data.abs().max()

                # Sync
                if self._stats_mode == "" and torch.cuda.device_count() > 1:
                    act_abs_max = AllReduce.apply(act_abs_max) * (1.0 / dist.get_world_size())

                self.beta_t = self.beta_t * self.beta
                self.act_max = self.act_max * self.beta + (act_abs_max * (1 - self.beta)) / (1 - self.beta_t)

                # print("self.running_stat:{}  act_max:{}".format(self.running_stat, self.act_max))

            quantized_input, scale_int_input = uniform_symmetric_quantizer_per_tensor(input, bits=self.quantization_bits,
                                                                     scale_bits=self.scale_bits, act_max=self.act_max)

        quantized_weight, scale_int_weight = uniform_symmetric_quantizer_per_channel(self.weight, bits=self.quantization_bits,
                                                                   scale_bits=self.scale_bits)

        if self.bias is not None:
            # quantization self.bias
            scale_int_bias = scale_int_input * scale_int_weight
            quantized_bias = uniform_symmetric_quantizer_per_channel(self.bias, bits=self.bias_bits, scale_bits=self.scale_bits, scale_int_bias=scale_int_bias)
        else:
            # not quantization self.bias
            quantized_bias = None

        return F.conv2d(quantized_input, quantized_weight, quantized_bias, self.stride, self.padding,
                        self.dilation, self.groups)


def replace_layer_by_unique_name(module, unique_name, layer):
    unique_names = unique_name.split(".")
    if len(unique_names) == 1:
        module._modules[unique_names[0]] = layer
    else:
        replace_layer_by_unique_name(
            module._modules[unique_names[0]],
            ".".join(unique_names[1:]),
            layer)


# replace model
def replace(model, quantization_bits=8, scale_bits=8, bias_bits=16, layer_layer_count=100000):
    count = 0
    for name, module in model.named_modules():

        # first layer and last layer quantized to 8 bit if quantization_bits <= 4

        if isinstance(module, nn.Conv2d):
            temp_conv = Conv2d_quantization(
                in_channels=module.in_channels,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                groups=module.groups,
                bias=(module.bias is not None),
                quantization_bits=quantization_bits,
                scale_bits=scale_bits,
                bias_bits=bias_bits,
                first_layer=(count == 0),
                last_layer=(count==layer_layer_count)
            )
            temp_conv.weight.data.copy_(module.weight.data)
            if module.bias is not None:
                temp_conv.bias.data.copy_(module.bias.data)
            replace_layer_by_unique_name(model, name, temp_conv)
            count += 1
    print("After replace:\n {}".format(model))
    return model


def freeze_model(model):
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):
            module.fix_activation_range()


def unfreeze_model(model):
    for name, module in model.named_modules():
        if isinstance(module, Conv2d_quantization):
            module.unfix_activation_range()


if __name__ == "__main__":
    # x = torch.rand(2, 3, 1, 1)
    # print('x.shape={}, x={}'.format(x.shape, x))
    # c_out, c_in, k, w = x.shape
    # x_reshape = x.reshape(c_out, -1)
    # maxv, _ = torch.max(torch.abs(x_reshape), 1)  # maxv shape = c_out*1
    # print('maxv.shape={}, maxv={}'.format(maxv.shape, maxv))

    # scale = get_scale_approximation(126.4, 8)
    # print(scale)

    x = torch.randint(1, 255, (1, 3, 2, 2)) + torch.rand((1, 3, 2, 2))
    print('x={}'.format(x))
    quantzied_x, scale_int = uniform_symmetric_quantizer_per_tensor(x, bits=8, scale_bits=8)
    print('quantzied_x={}'.format(quantzied_x))

    # x = torch.randint(1, 255, (3, 1, 1, 1))
    # print('x={}'.format(x))
    # y = torch.randint(1, 100, (3, 1, 1, 1))
    # print('y={}'.format(y))
    # z = torch.min(x ,y)
    # print('z={}'.format(z))

    # x = torch.randint(1, 255, (3, 1, 2, 1)) + torch.rand((3, 1, 2, 1))
    # # print('x={}'.format(x))
    # print('x.shape={},\nx={}'.format(x.shape, x))
    # quantzied_x, scale_int = uniform_symmetric_quantizer_per_channel(x, bits=8, scale_bits=8)
    # print('quantzied_x.shape={}, \nquantzied_x={}'.format(quantzied_x.shape, quantzied_x))