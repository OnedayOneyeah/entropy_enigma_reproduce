import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F

from models.param import load_model_and_optimizer, copy_model_and_optimizer, configure_model, collect_params
from models.optim import setup_optimizer

class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x, if_adapt=True, counter=None, if_vis=False):
        if self.episodic:
            self.reset()

        if if_adapt:
            #print("adaptating..")
            for _ in range(self.steps):
                outputs = forward_and_adapt(x, self.model, self.optimizer)
                # print("output: {}".format(F.softmax(outputs)[0].detach().cpu().numpy()))
                # print("sort: {}".format(torch.sort(F.softmax(outputs)[0])[0].detach().cpu().numpy()))
        else:
            #print("no adaptation")
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.

    Measure entropy of the model prediction, take gradients, and update params.
    """
    # forward
    outputs = model(x)
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return outputs

def setup_tent(model, cfg, logger):
    """Set up tent adaptation.
    Configure the model for training + feature modulation by batch statistics,
    collect the parameters for feature modulation by gradient optimization,
    set up the optimizer, and then tent the model.
    """
    model = configure_model(model, 
                            ada_param=['bn'])
    params, param_names = collect_params(model,
                                         ada_param=['bn'],
                                         logger=logger)
    optimizer = setup_optimizer(params, cfg, logger)
    tent_model = Tent(model, optimizer,
                           steps=1,
                           episodic=False)
    logger.info(f"model for adaptation: %s", model)
    logger.info(f"params for adaptation: %s", param_names)
    logger.info(f"optimizer for adaptation: %s", optimizer)
    return tent_model

    
