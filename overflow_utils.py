import torch
import torch.distributed as dist
from torch.autograd import Function

import tflite_quantize_OAQ as tflite


# overflow aware quantization
class AllReduce_overflow(Function):
    @staticmethod
    def forward(ctx, input):
        input_list = [torch.zeros_like(input) for k in range(dist.get_world_size())]
        # Use allgather instead of allreduce since I don't trust in-place operations ..
        dist.all_gather(input_list, input, async_op=False)
        inputs = torch.stack(input_list, dim=0)
        return torch.sum(inputs, dim=0)


def calculate_No(device, model, oaq_conv_result, conv_accumulator_bits, logger):
    logger.info("calculate No: start!")
    logger.info('len(oaq_conv_result)={}'.format(len(oaq_conv_result)))

    # conv_accumulator_bits: [min, max], sign=True
    min = - 2 ** (conv_accumulator_bits - 1) + 1
    max = 2 ** (conv_accumulator_bits - 1) - 1

    index = 0
    No = torch.zeros(len(oaq_conv_result)).to(device)   # nx1, n: the number of conv layer
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            # oaq_conv_result[index]: batch*C_out*h*w
            # layer.scale_int_weight: [C_out]
            scale = (layer.scale_int_input * layer.scale_int_weight).unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(oaq_conv_result[index])
            oaq_conv_result[index] = torch.round(oaq_conv_result[index] / scale)
            down_overflow_index = oaq_conv_result[index] < min
            up_overflow_index = oaq_conv_result[index] > max
            No[index] = (torch.sum(down_overflow_index) + torch.sum(up_overflow_index)).to(device)
            index += 1

    if index != len(oaq_conv_result):
        assert False, logger.info('Conv2d_quantization number != len(oaq_conv_result)')
    return No


def update_alpha(device, model, No, iteration_batch_size, lr_max, lr_curr, logger):
    logger.info("update alpha: start!")

    # merge No from every GPU
    logger.info('before merge, No={}'.format(No))
    No = AllReduce_overflow.apply(No)
    logger.info('After merge, No={}'.format(No))

    index = 0
    for name, layer in model.named_modules():
        if isinstance(layer, tflite.Conv2d_quantization):
            logger.info('index = {}'.format(index))
            logger.info('Before update, activation_alpha={}, weight_alpha={}'.format(layer.activation_alpha, layer.weight_alpha))
            # logger.info('No[{}]={}, iteration_batch_size={}, lr_max={}, lr_curr={}'.format(index, No[index], iteration_batch_size, lr_max, lr_curr))
            if No[index] > 0:
                # v1: better
                update_value = torch.min((lr_curr * torch.log(No[index] / iteration_batch_size)), torch.Tensor([lr_max])[0].to(device))
                # v2
                # update_value = torch.min((lr_curr * torch.log(No[index])), torch.Tensor([lr_max])[0].to(device))
                layer.activation_alpha += update_value
                layer.weight_alpha += update_value

            elif No[index] == 0:
                lr_curr_gpu = torch.Tensor([lr_curr])[0].to(device)
                layer.activation_alpha -= lr_curr_gpu
                layer.weight_alpha -= lr_curr_gpu

            else:
                assert False, logger.info('No[{}] ={} impossible !!!'.format(index, No[index]))
            index += 1
            logger.info('After update, activation_alpha={}, weight_alpha={}'.format(layer.activation_alpha, layer.weight_alpha))