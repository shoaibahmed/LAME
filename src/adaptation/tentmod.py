import torch
import torch.jit
import logging
from typing import List, Dict
import time

import os
import glob
import numpy as np
import cv2

import src.data.utils
from src.data.datasets.classes.imagenet import INDEX2IDNAME, ID2INDEX

from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY

from ..contrib import get_imagenet_examples

__all__ = ["TentMod"]

logger = logging.getLogger(__name__)


@ADAPTER_REGISTRY.register()
class TentMod(AdaptiveMethod):
    """
    Tent method (https://arxiv.org/abs/2006.10726), but combined with a mode collapse solution.
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
        self.initialize_examples()

    def clone_optim_params(self):
        print("Cloning optimization parameters...")
        # Expected structure : List (of param dicts) -> Dict (of group of parameters) -> List (of parameters)
        param_groups = self.optimizer.param_groups
        print([param_dict.keys() for param_dict in param_groups])
        print([[x.shape for x in param_group['params']] for param_group in param_groups])
        # self.param_groups = [{k: [x.clone().detach() for x in param_group[k]] for k in param_group.keys()} for param_group in param_groups]
        self.param_groups = [{'params': [x.clone().detach() for x in param_group['params']]} for param_group in param_groups]

    def initialize_examples(self, use_model_inversion=False):
        if hasattr(self, 'extra_examples'):
            return
        print("Initializing model examples...")

        self.bs = self.cfg.ADAPTATION.NUM_GENERATED_EXAMPLES
        if use_model_inversion:
            print("Using model inversion to compute examples...")
            self.extra_examples = get_imagenet_examples(self.model, bs=self.bs)
            print("Generated examples shape:", len(self.extra_examples), "/", self.extra_examples[0]['image'].shape)
        else:
            print("Using a selected number of examples from the actual training set...")
            img_files = glob.glob("/mnt/sas/Datasets/ilsvrc12/small_train_10_imgs/*/*.JPEG")
            image_list = []
            input_format = self.input_format # self.model.input_format
            assert input_format in ["RGB", "BGR"], input_format
            print("Using input format:", input_format)
            
            for img_file in img_files:
                img = cv2.imread(img_file)  # Loads image in BGR format
                assert img is not None
                
                # Resize image to 224 x 224
                img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
                
                # if self.model.normalize_input:  # Model uses RGB inputs rather than BGR
                if input_format == "RGB":  # Model uses RGB inputs rather than BGR
                    print("Converting image to RGB file format...")
                    img = img[:, :, ::-1]
                
                instances = []
                dir_name = img_file.split(os.sep)[-2]
                class_id = ID2INDEX[dir_name]
                cat_index = class_id
                cat_id, cat_name = INDEX2IDNAME[cat_index]
                instances.append({"category_id": cat_id, "supercategory_id": cat_id, 
                            "supercategory_index": cat_index, "category_index": cat_index,
                            "category_name": cat_name, "supercategory_name": cat_name})
                # r["annotations"] = instances
                image_shape = (img.shape[1], img.shape[0], img.shape[2])  # Format it as WHC instead of HWC
                instances = src.data.utils.annotations_to_instances(instances, image_shape)

                image_list.append({'image': img, 'instances': instances})
            self.extra_examples = image_list

        # Compute the feature so as to compute the alignment loss
        with torch.no_grad():
            if self.use_eval_mode:
                print("!! Using eval mode for computing probe features...")
                is_training = self.model.training  # Obtained from the module class
                self.model.eval()  # Put the model in eval mode
            
            if self.bs == self.extra_examples:  # Generated examples
                output_dict = self.model(self.extra_examples)
                self.precomputed_features_ex = output_dict['features'].detach()
                self.precomputed_logits_ex = output_dict['logits'].detach()
            else:
                n_batches = int(np.ceil(float(len(self.extra_examples)) / self.bs))
                print(f"Num examples: {len(self.extra_examples)} / Batch size: {self.bs} / Num batches: {n_batches}")
                iterator = 0
                self.precomputed_features_ex = []
                self.precomputed_logits_ex = []
                
                for i in range(n_batches):
                    output_dict = self.model(self.extra_examples[iterator:(iterator+self.bs)])
                    self.precomputed_features_ex.append(output_dict['features'].detach())
                    self.precomputed_logits_ex.append(output_dict['logits'].detach())
                    iterator += len(self.precomputed_logits_ex[-1])
                assert iterator == len(self.extra_examples), f"{iterator} != {len(self.extra_examples)}"
                
                self.precomputed_features_ex = torch.cat(self.precomputed_features_ex, dim=0)
                self.precomputed_logits_ex = torch.cat(self.precomputed_logits_ex, dim=0)
                assert len(self.precomputed_features_ex) == len(self.extra_examples)
                assert len(self.precomputed_logits_ex) == len(self.extra_examples)
            
            if self.use_eval_mode:
                self.model.train(mode=is_training)  # Set the model back into the same training mode

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):
        t0 = time.time()
        # Compute the probs on the given set of examples
        probas = self.model(batched_inputs)['probas']
        
        if not self.use_param_alignment:
            # Compute the feature alignment loss on a randomly selected number of examples
            selected_ex = np.random.choice(np.arange(len(self.extra_examples)), size=self.bs, replace=False)
            extra_examples = [self.extra_examples[i] for i in selected_ex]
            target_features = torch.stack([self.precomputed_features_ex[i] for i in selected_ex], dim=0)
            target_logits = torch.stack([self.precomputed_logits_ex[i] for i in selected_ex], dim=0)
            
            if self.use_eval_mode:
                is_training = self.model.training  # Obtained from the module class
                self.model.eval()  # Put the model in eval mode
            
            output_dict = self.model(extra_examples)
            features_ex = output_dict['features']
            logits_ex = output_dict['logits']
            
            if self.use_eval_mode:
                self.model.train(mode=is_training)  # Set the model back into the same training mode
        
        self.metric_hook.scalar_dic["forward_time"].append(time.time() - t0)
        t1 = time.time()

        log_probas = torch.log(probas + 1e-10)
        entropy = -(probas * log_probas).sum(-1).mean(0)
        
        if self.use_param_alignment:
            params_alignment_loss = 0.
            counter = 0
            k = 'params'
            for i in range(len(self.param_groups)):
                for j in range(len(self.param_groups[i][k])):
                    params_alignment_loss = params_alignment_loss + self.alignment_criterion(self.optimizer.param_groups[i][k][j], self.param_groups[i][k][j])
                    counter += 1
            params_alignment_loss = params_alignment_loss / counter

            loss = entropy + self.cfg.ADAPTATION.LAMBDA_ALIGNMENT * params_alignment_loss
            print(f"Loss stats / Entropy: {entropy:.4f} / Params diff loss: {params_alignment_loss:.8f} / Total: {loss:.4f}", flush=True)
        
        else:    
            # Include the second loss term
            feature_alignment_loss = self.alignment_criterion(features_ex, target_features)
            logit_alignment_loss = self.alignment_criterion(logits_ex, target_logits)

            alignment_loss = 0.5 * feature_alignment_loss + 0.5 * logit_alignment_loss
            loss = entropy + self.cfg.ADAPTATION.LAMBDA_ALIGNMENT * alignment_loss

            print(f"Loss stats / Entropy: {entropy:.4f} / Feature alignment: {feature_alignment_loss:.4f} / Logit alignment: {logit_alignment_loss:.4f} / Total: {loss:.4f}", flush=True)

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[union-attr]
        self.optimizer.step()

        self.metric_hook.scalar_dic["optimization_time"].append(time.time() - t1)
        self.metric_hook.scalar_dic['full_loss'].append(loss.item())
