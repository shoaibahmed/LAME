import torch
import torch.jit
import logging
from typing import List, Dict
import time

from .adaptive import AdaptiveMethod
from .build import ADAPTER_REGISTRY

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

        self.lambda_alignment = 0.1

        # Compute the extra examples here
        # TODO: Add a way to infer these examples
        self.extra_examples = None

        # Compute the feature so as to compute the alignment loss
        with torch.no_grad():
            self.model.eval()
            self.precomputed_features_ex = self.model(self.extra_examples)['features']
            self.model.train()
        self.feature_criterion = torch.nn.MSELoss()

    def run_optim_step(self, batched_inputs: List[Dict[str, torch.Tensor]], **kwargs):

        t0 = time.time()
        # Compute the probs on the given set of examples
        probas = self.model(batched_inputs)['probas']
        
        # Compute the feature alignment loss on the given number of examples
        features_ex = self.model(self.extra_examples)['features']

        self.metric_hook.scalar_dic["forward_time"].append(time.time() - t0)
        t1 = time.time()

        log_probas = torch.log(probas + 1e-10)
        entropy = -(probas * log_probas).sum(-1).mean(0)
        loss = entropy

        # Include the second loss term
        alignment_loss = self.feature_criterion(self.precomputed_features_ex, features_ex)
        loss = loss + self.lambda_alignment * alignment_loss

        self.optimizer.zero_grad()
        loss.backward()  # type: ignore[union-attr]
        self.optimizer.step()

        self.metric_hook.scalar_dic["optimization_time"].append(time.time() - t1)
        self.metric_hook.scalar_dic['full_loss'].append(loss.item())