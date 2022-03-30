import os
import pickle

import torch
from torch import nn
from .deepinversion import DeepInversionClass


def get_probs(net, image):
    if not net.normalize_input:
        image = torch.clamp(image * 255, 0, 255)
    if net.standardize_input:
        image = (image - net.pixel_mean) / net.pixel_std
    return net(image)['probas']


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

    model.eval()
    with torch.no_grad():
        output = get_probs(model, input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    model.train()
    print("Verifier accuracy: ", prec1.item())


def get_imagenet_examples(net, bs=256):
    # Check if the pickle file already contains the batch information
    cache_data_path = "./final_images/%s/cache.pkl"%exp_name
    if os.path.exists(cache_data_path):
        print("Loading data from cache file...")
        with open(cache_data_path, "rb") as f:
            image_list = pickle.load(f)
        if len(image_list) == bs:
            return image_list
        else:
            print(f"Warning: the cache file contains a different number of examples ({len(image_list)} instead of {bs}). Regenerating images...")

    exp_name = "tentmod_test"
    # final images will be stored here:
    adi_data_path = "./final_images/%s"%exp_name
    # temporal data and generations will be stored here
    exp_name = "generations/%s"%exp_name

    setting_id = 0  # settings for optimization: 0 - multi resolution, 1 - 2k iterations, 2 - 20k iterations
    fp16 = False
    jitter = 30

    parameters = dict()
    parameters["resolution"] = 224
    parameters["random_label"] = True
    parameters["start_noise"] = True
    parameters["detach_student"] = False
    parameters["do_flip"] = True
    parameters["store_best_images"] = True

    criterion = nn.CrossEntropyLoss()

    coefficients = dict()
    coefficients["adi_scale"] = 0.0

    if net.normalize_input:
        # Input range between [0, 1]
        coefficients["r_feature"] = 0.01
        coefficients["first_bn_multiplier"] = 10.0
        coefficients["tv_l1"] = 0.0
        coefficients["tv_l2"] = 0.0001
        coefficients["l2"] = 0.00001
        coefficients["lr"] = 0.25
        coefficients["main_loss_multiplier"] = 1.0
    else:
        # Input range between [0, 255]
        coefficients["r_feature"] = 0.001  # Factor of 10 change
        coefficients["first_bn_multiplier"] = 10 / 255.  # 10 / 255. -- the scale of inputs has changed
        coefficients["tv_l1"] = 0.0
        coefficients["tv_l2"] = 0.0001
        coefficients["l2"] = 0.00001
        coefficients["lr"] = 0.025  # Reduced by a factor of 10
        coefficients["main_loss_multiplier"] = 10.0

    network_output_function = lambda x: x['logits']

    # check accuracy of verifier
    verifier = False  # Wrong input range -- so will always give wrong predictions
    net.eval()
    if verifier:
        hook_for_display = lambda x,y: validate_one(x, y, net)
    else:
        hook_for_display = None

    DeepInversionEngine = DeepInversionClass(net_teacher=net,
                                             final_data_path=adi_data_path,
                                             path=exp_name,
                                             parameters=parameters,
                                             setting_id=setting_id,
                                             bs = bs,
                                             use_fp16 = fp16,
                                             jitter = jitter,
                                             criterion=criterion,
                                             coefficients = coefficients,
                                             network_output_function = network_output_function,
                                             hook_for_display = hook_for_display)
    image_list = DeepInversionEngine.generate_batch()

    # Write data to the cache file
    with open(cache_data_path, "wb") as f:
        pickle.dump(image_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        print("Data saved to cache file...")

    return image_list
