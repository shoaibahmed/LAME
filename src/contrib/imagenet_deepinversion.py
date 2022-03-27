import argparse

import torch
from torch import nn
from deepinversion import DeepInversionClass


def validate_one(input, target, model):
    """Perform validation on the validation set"""

    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)['probas']
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

    print("Verifier accuracy: ", prec1.item())


def get_imagenet_examples(net, bs=256):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    exp_name = args.exp_name
    # final images will be stored here:
    adi_data_path = "./final_images/%s"%exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s"%exp_name

    args.iterations = 2000
    args.start_noise = True
    args.setting_id = 0  # Multi-scale
    args.fp16 = False

    args.resolution = 224
    # bs = 256
    jitter = 30

    parameters = dict()
    parameters["resolution"] = 224
    parameters["random_label"] = False
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True
    parameters["store_best_images"] = True

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["r_feature"] = 0.01
    coefficients["first_bn_multiplier"] = 10.
    coefficients["tv_l1"] = 0.0
    coefficients["tv_l2"] = 0.0001
    coefficients["l2"] = 0.00001
    coefficients["lr"] = 0.25
    coefficients["main_loss_multiplier"] = 1.0
    coefficients["adi_scale"] = 0.0

    network_output_function = lambda x: x['probas']

    # check accuracy of verifier
    net.eval()
    if args.verifier:
        hook_for_display = lambda x,y: validate_one(x, y, net)
    else:
        hook_for_display = None

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             final_data_path=adi_data_path,
                                             path=exp_name,
                                             parameters=parameters,
                                             setting_id=args.setting_id,
                                             bs = bs,
                                             use_fp16 = args.fp16,
                                             jitter = jitter,
                                             criterion=criterion,
                                             coefficients = coefficients,
                                             network_output_function = network_output_function,
                                             hook_for_display = hook_for_display)
    image_list = DeepInversionEngine.generate_batch()
    return image_list
