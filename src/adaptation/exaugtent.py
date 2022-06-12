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

from tqdm import tqdm
from contextlib import ExitStack, contextmanager
from src.utils.events import EventStorage

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

        self.augmentor = ImageAugmentator()
        self.num_augs = 16
        self.num_optim_steps = 10
        
        self.alignment_criterion = torch.nn.MSELoss()
        self.clone_optim_params()

    def clone_optim_params(self):
        print("Cloning optimization parameters...")
        # Expected structure : List (of param dicts) -> Dict (of group of parameters) -> List (of parameters)
        param_groups = self.optimizer.param_groups
        print([param_dict.keys() for param_dict in param_groups])
        print([len(param_group['params']) for param_group in param_groups], [[x.shape for x in param_group['params']] for param_group in param_groups])
        self.param_groups = [{'params': [x.clone().detach() for x in param_group['params']]} for param_group in param_groups]
    
    def reset_model_params(self):
        # k = 'params'
        # for i in range(len(self.param_groups)):
        #     for j in range(len(self.param_groups[i][k])):
        #         self.optimizer.param_groups[i][k][j] = self.param_groups[i][k][j].clone().detach()
        # self.optimizer.param_groups = copy.deepcopy(self.param_groups)
        self.reset_model_optim()
    
    def run_episode(self, loader: torch.utils.data.DataLoader) -> EventStorage:
        """
        Loader contains all the samples in one run, and yields them by batches.
        """
        self.reset_model_optim()
        max_inner_iters = self.cfg.ADAPTATION.MAX_BATCH_PER_EPISODE
        with EventStorage(0) as local_storage:

            for _ in range(self.steps):
                batch_limitator = range(max_inner_iters)
                bar = tqdm(loader, total=min(len(loader), max_inner_iters))
                bar.set_description(f"Running optimization steps {'online' if self.online else 'offline'}")
                for i, (batched_inputs, indexes) in zip(batch_limitator, bar):

                    # --- Optimization part ---

                    # self.run_optim_step(batched_inputs, indexes=indexes, loader=loader, batch_index=i)

                    # --- Evaluation part ---

                    if self.online:
                        # with ExitStack() as stack:
                        #     stack.enter_context(inference_context(self.model))
                        #     stack.enter_context(torch.no_grad())
                        self.before_step()
                        t0 = time.time()
                        outputs = self.run_step(batched_inputs)
                        self.metric_hook.scalar_dic["inference_time"].append(time.time() - t0)
                        self.after_step(batched_inputs, outputs)

        return local_storage
    
    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):
        raise RuntimeError("No optimization step applicable...")  # Do nothing at training time
    
    def run_step(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        # Compute the probs on the given set of examples
        # Roll back the parameters to the original values before optimizing anything
        
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
            
            print("Computing param alignment loss before adaptation:", self.compute_param_diff())
            
            for j in range(self.num_optim_steps):
                probas = self.model(current_example_batch)['probas']
                
                # Optimize the model over these samples
                log_probas = torch.log(probas + 1e-10)
                entropy = -(probas * log_probas).sum(-1).mean(0)
                
                self.optimizer.zero_grad()
                entropy.backward()
                self.optimizer.step()
                
                print(f"Computing param alignment loss after update # {j+1}:", self.compute_param_diff())
            
            print(f"Inference step stats for example # {i+1} / Entropy: {entropy:.4f}", flush=True)
            
            # Produce the predictions
            out = self.model([current_example])
            prob_list.append(out['probas'].detach())
            logit_list.append(out['logits'].detach())
            features_list.append(out['features'].detach())
            
            # Reset the model
            self.reset_model_params()
            
            print("Computing param alignment loss after resetting model weights:", self.compute_param_diff())
            exit()
        
        probas = torch.cat(prob_list, dim=0)
        logits = torch.cat(logit_list, dim=0)
        features = torch.cat(features_list, dim=0)
        
        final_output = self.model.format_result(batched_inputs, logits, probas, features)
        return final_output

    def compute_param_diff(self):
        k = 'params'
        counter = 0
        params_alignment_loss = 0.
        
        for i in range(len(self.param_groups)):
            for j in range(len(self.param_groups[i][k])):
                params_alignment_loss = params_alignment_loss + self.alignment_criterion(self.optimizer.param_groups[i][k][j], self.param_groups[i][k][j])
                counter += 1
        params_alignment_loss = params_alignment_loss / counter
        return float(params_alignment_loss)

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
