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
    TENT~\cite{wang2020tent} minimizes the entropy of the predictions.
    Concretely, let us denote the original dataset which was used to pretrain the model as $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^{N}$ where $N$ is the number of examples.
    The aim is to adapt the model to a new dataset ($\mathcal{D}' = \{(x_i)\}_{i=1}^{n}$ where $n$ is the number of new examples) at test-time.
    The original TENT objective can be written as:

    \begin{equation*}
    \begin{gathered}
        f(\mathbf{x}, \mathcal{W}_{BN}^{'}) = \Psi \bigg( \Phi(\mathbf{x}; \mathcal{W}_F \cup \mathcal{W}_{BN}^{'}); \mathcal{W}_C \bigg) \\
        \mathcal{W}_{BN}^{*} = \argmin{\mathcal{W}_{BN}^{'}} - \frac{1}{\abs{\mathcal{D}'}} \sum_{\mathbf{x} \in \mathcal{D}'} \sum_{c=1}^{C} f(\mathbf{x}, \mathcal{W}_{BN}^{'})_c \log f(\mathbf{x}, \mathcal{W}_{BN}^{'})_c
    \end{gathered}
    \end{equation*}

    \noindent where $\Phi$ represents the features returned by the model (assuming it to be the penultimate layer of the network) with pretrained parameters $\mathcal{W}_F$, $\Psi$ represents the classification head with pretrained parameters $\mathcal{W}_C$, and $C$ represents the number of output classes.
    The feature extractor also includes pretrained batch-norm parameters $\mathcal{W}_{BN}$ that includes mean, variance, scale and bias.
    We discard these parameters and optimize the model over a new set of batch-norm parameters $\mathcal{W}_{BN}^{'}$.
    Since the classification head only contains a single dense layer, all the batch-norm parameters are located in the feature extraction layers.
    The optimization process returns an optimized set of batch-norm parameters $\mathcal{W}_{BN}^{*}$ that can be plugged into the network for final inference.

    Using this simple formulation results in model drift when considering non-IID settings as highlighted in~\cite{boudiaf2022parameter}.
    One of the simplest ways to circumvent this problem is to train the model jointly on the original dataset and well as the new dataset (offline setting) or include an alignment term based on the original dataset when computing the batch-norm parameters.
    Two most prominent reasons why they disregard access to the training dataset includes (i) their proprietary nature as well as (ii) large computational budget required to consider these large datasets during finetuning.
    Disregarding privacy concerns, online adaptation by including large-scale datasets within the finetuning step is computationally prohibitive.
    Therefore, a natural question is: \textit{can we use a very small number of carefully computed proxy samples that can be used to minimize the model drift towards degenerate solutions?}

    We consider three potential ways to generate examples that can be further used to train the model.
    \begin{enumerate}
        \item Use procedural noise such as fractals~\cite{kataoka2020pre} or the one that can be readily generated through untrained GANs~\cite{baradad2021learning}. These fractals can be readily generated, and requires no knowledge regarding the dataset used for pretraining.
        \item Condense the pretraining dataset itself into a small number of synthetically optimized examples. Approaches such as gradient-matching have been successfully leveraged in the past to obtain a small number of (optimized) examples that can be used to train the model such that the model, when trained on this subset of examples, obtains similar accuracy as that obtained by training on the entire dataset~\cite{zhao2020condensation} (\textcolor{red}{these approaches are not too prominent for large-scale datasets such as ImageNet}).
        \item Generate a number of examples from the pretrained model itself. This can be done using data-free approaches that generate examples by matching the statistics of the activations generated from random examples that are being optimized to the statistics stored in the pretrained model in the form of batch-norm parameters~\cite{yin2020dreaming}.
    \end{enumerate}
    % A simple way is to use procedural noise such as fractals~\cite{kataoka2020pre} or the one that can be readily generated through untrained GANs~\cite{baradad2021learning}.
    % An alternate way of obtaining a small set of examples to avoid degenerate solutions is to condense the dataset itself in a small set of synthetically optimized examples.
    % Approaches such as gradient-matching have been successfully leveraged in the past to obtain a small number of (optimized) examples that can be used to train the model such that the model, when trained on this subset of examples, obtains similar accuracy as that obtained by training on the entire dataset~\cite{zhao2020condensation} (\textcolor{red}{these approaches are not too prominent for large-scale datasets such as ImageNet}).
    % Data free approaches that attempts to generate examples from a pretrained model using model inversion techniques can also be employed for this purpose~\cite{yin2020dreaming} to generate a limited number of examples using the pretrained model itself.

    The proposed approach is based on the premise that aligning the model on this procedural noise i.e. constraining the model to produce the same feature embeddings before and after adaptation on these generated/selected inputs should avoid model drift such that the model results in degenerate solutions.

    Now, let us assume access to another dataset ($\mathcal{D}^C = \{(x_i)\}_{i=1}^{k}$ where $k$ is the number of examples and $k << N$) which can either be a condensed form of the original dataset~\cite{zhao2020condensation} or procedurally generated noise~\cite{kataoka2020pre,baradad2021learning}.
    Notice that we only assume access to inputs without access to their corresponding labels.
    In order to avoid degenerate solutions, we can add a constraint on the optimization problem to remain faithful to the original task:

    \begin{equation*}
    \begin{gathered}
        g(\mathbf{x}, \mathcal{W}_{BN}^{'}) = \norm{\Phi(\mathbf{x}; \mathcal{W}_F \cup \mathcal{W}_{BN}^{'}) - \Phi(\mathbf{x}; \mathcal{W}_F \cup \mathcal{W}_{BN})}_2^2 \\
        \mathcal{W}_{BN}^{*} = \argmin{\mathcal{W}_{BN}^{'}} - \frac{1}{\abs{\mathcal{D}'}} \sum_{\mathbf{x} \in \mathcal{D}'} \sum_{c=1}^{C} f(\mathbf{x}, \mathcal{W}_{BN}^{'})_c \log f(\mathbf{x}, \mathcal{W}_{BN}^{'})_c \\ + \lambda \frac{1}{\abs{\mathcal{D}^{C}}} \sum_{\mathbf{x} \in \mathcal{D}^{C}} g(\mathbf{x}, \mathcal{W}_{BN}^{'})
    \end{gathered}
    \end{equation*}

    \noindent where $g$ captures the distance between the features returned by the pretrained batch-norm parameters and the new set of batch-norm parameters.
    Adding this loss term ensures that the model remains faithful to the original task when finetuning.
    This introduces a new hyperparameter into the system ($\lambda$).
    This $\lambda$ defines the trade-off between minimizing the entropy on the target dataset and minimizing the deviation from the source distribution.
    \textcolor{red}{I assume that a simple value of $\lambda$ such as 1 might suffice for a range of different problems in practice.}

    """

    def __init__(self, cfg, args, **kwargs):
        """
        Args:
            cfg (CfgNode):
        """
        super().__init__(cfg, args, **kwargs)

        self.alignment_criterion = torch.nn.MSELoss()
        self.use_param_alignment = self.cfg.ADAPTATION.USE_PARAM_ALIGNMENT
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
            img_files = glob.glob("/mnt/sas/Datasets/ilsvrc12/small_train_2_imgs/*/*.JPEG")
            image_list = []
            
            for img_file in img_files:
                img = cv2.imread(img_file)  # Loads image in BGR format
                assert img is not None
                
                if self.model.normalize_input:  # Model uses RGB inputs rather than BGR
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
            is_training = self.model.training  # Obtained from the module class
            self.model.eval()  # Put the model in eval mode
            if self.bs == self.extra_examples:  # Generated examples
                output_dict = self.model(self.extra_examples)
                self.precomputed_features_ex = output_dict['features'].detach()
                self.precomputed_logits_ex = output_dict['logits'].detach()
            else:
                n_batches = np.ceil(self.extra_examples / self.bs)
                print(f"Num examples: {self.extra_examples} / Batch size: {self.bs} / Num batches: {n_batches}")
                iterator = 0
                self.precomputed_features_ex = []
                self.precomputed_logits_ex = []
                
                for i in range(n_batches):
                    output_dict = self.model(self.extra_examples[iterator:(iterator+self.bs)])
                    self.precomputed_features_ex.append(output_dict['features'].detach())
                    self.precomputed_logits_ex.append(output_dict['logits'].detach())
                    iterator += len(self.precomputed_logits_ex[-1])
                assert iterator == self.extra_examples
                
                self.precomputed_features_ex = torch.cat(self.precomputed_features_ex, dim=0)
                self.precomputed_logits_ex = torch.cat(self.precomputed_logits_ex, dim=0)
                assert len(self.precomputed_features_ex) == len(self.extra_examples)
                assert len(self.precomputed_logits_ex) == len(self.extra_examples)
            
            self.model.train(mode=is_training)  # Set the model back into the same training mode

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):
        t0 = time.time()
        # Compute the probs on the given set of examples
        probas = self.model(batched_inputs)['probas']
        
        if not self.use_param_alignment:
            # Compute the feature alignment loss on a randomly selected number of examples
            selected_ex = np.random.choice(np.arange(len(self.extra_examples)), size=self.bs, replace=False)
            extra_examples = [self.extra_examples[i] for i in selected_ex]
            target_features = [self.precomputed_features_ex[i] for i in selected_ex]
            target_logits = [self.precomputed_logits_ex[i] for i in selected_ex]
            
            output_dict = self.model(extra_examples)
            features_ex = output_dict['features']
            logits_ex = output_dict['logits']

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
