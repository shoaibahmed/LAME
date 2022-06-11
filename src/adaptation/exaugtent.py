import torch
import torch.jit
import logging
from typing import List, Dict
import time
import copy

import numpy as np

from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY

from ..imagenet_c_aug import ImageAugmentator

__all__ = ["ExAugTent"]

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register()
class ExAugTent(AdaptiveMethod):
    """
    Tent method (https://arxiv.org/abs/2006.10726), but with per-example optimization.
    """

    def __init__(self, cfg, args, **kwargs):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg, args, **kwargs)

        self.alignment_criterion = torch.nn.MSELoss()
        self.use_param_alignment = self.cfg.ADAPTATION.USE_PARAM_ALIGNMENT
        self.use_eval_mode = self.cfg.ADAPTATION.USE_EVAL_MODE
        self.input_format = self.cfg.INPUT.FORMAT
        self.clone_optim_params()
        
        self.augmentor = ImageAugmentator()
        self.num_augs = 16
        self.num_optim_steps = 10

    def clone_optim_params(self):
        # print("Cloning optimization parameters...")
        # # Expected structure : List (of param dicts) -> Dict (of group of parameters) -> List (of parameters)
        # param_groups = self.optimizer.param_groups
        # print([param_dict.keys() for param_dict in param_groups])
        # print([[x.shape for x in param_group['params']] for param_group in param_groups])
        # self.param_groups = [{'params': [x.clone().detach() for x in param_group['params']]} for param_group in param_groups]
        self.param_groups = copy.deepcopy(self.optimizer.param_groups)
    
    def reset_model_params(self):
        self.optimizer.param_groups = self.param_groups

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):
        pass  # Do nothing at training time
    
    def visualize_images(self, imgs, output_file=None):
        import matplotlib.pyplot as plt
        
        plot_size = 3
        plot_rows = 3
        num_plots_per_row = 3
        fig, ax = plt.subplots(plot_rows, num_plots_per_row, figsize=(plot_size * num_plots_per_row, plot_size * plot_rows), sharex=True, sharey=True)

        for idx in range(len(imgs)):
            ax[idx // num_plots_per_row, idx % num_plots_per_row].imshow(imgs[idx]['image'])
            # ax[idx // num_plots_per_row, idx % num_plots_per_row].set_title(y[idx])

            if idx == plot_rows * num_plots_per_row - 1:
                break

        for a in ax.ravel():
            a.set_axis_off()

            # Turn off tick labels
            a.set_yticklabels([])
            a.set_xticklabels([])

        fig.tight_layout()
        if output_file is not None:
            fig.savefig(output_file, bbox_inches=0.0, pad_inches=0)
        plt.close()
    
    def run_step(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        # Compute the probs on the given set of examples
        # Roll back the parameters to the original values before optimizing anything
        torch.set_grad_enabled(True)
        
        prob_list = []
        logit_list = []
        features_list = []
        
        # Reset the model
        self.reset_model_params()
        
        for i in range(len(batched_inputs)):
            # Add example augmentations
            current_example = batched_inputs[i]
            current_example_batch = [current_example] + [{'image': self.augmentor.apply_aug(current_example['image']), 'instances': current_example['instances']} for _ in range(self.num_augs)]
            
            # print("Running inference step...")
            # print(type(current_example_batch), [type(current_example_batch[i]) for i in  range(len(current_example_batch))], [type(current_example_batch[i]['image']) for i in range(len(current_example_batch))])
            # print("Shape:", [batched_inputs[i]['image'].shape for i in range(len(batched_inputs))])
            # print("Dtype:", [batched_inputs[i]['image'].dtype for i in range(len(batched_inputs))])
            # print("Min:", [batched_inputs[i]['image'].min() for i in range(len(batched_inputs))])
            # print("Mean:", [batched_inputs[i]['image'].mean() for i in range(len(batched_inputs))])
            # print("Max:", [batched_inputs[i]['image'].max() for i in range(len(batched_inputs))])
            # self.visualize_images(current_example_batch, output_file="aug_test.png")
            # self.visualize_images(batched_inputs, output_file="original_batch_test.png")
            
            for _ in range(self.num_optim_steps):
                probas = self.model(current_example_batch)['probas']
                
                # Optimize the model over these samples
                log_probas = torch.log(probas + 1e-10)
                entropy = -(probas * log_probas).sum(-1).mean(0)
                
                self.optimizer.zero_grad()
                entropy.backward()  # type: ignore[union-attr]
                self.optimizer.step()
            
            print(f"Inference step stats for example # {i+1} / Entropy: {entropy:.4f}", flush=True)
            
            # Produce the predictions
            out = self.model([current_example])
            prob_list.append(out['probas'].detach())
            logit_list.append(out['logits'].detach())
            features_list.append(out['features'].detach())
            
            # Reset the model
            self.reset_model_params()
        
        probas = torch.cat(prob_list, dim=0)
        logits = torch.cat(logit_list, dim=0)
        features = torch.cat(features_list, dim=0)
        torch.set_grad_enabled(False)
        
        final_output = self.model.format_result(batched_inputs, logits, probas, features)
        return final_output
        
        # out = self.model(batched_inputs)
        # logits = out["logits"]
        # probas = out["probas"]
        # features = out["features"]
