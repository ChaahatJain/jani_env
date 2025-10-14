import copy
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torch

class HookSimpleMLPLayer:
    """ Simple hook for a single layer
    """
    def __init__(self, model, layer_name):       
        self.layer_forward = []
        self.layer_backward = []
        
        for name, module in model.named_modules():
            if layer_name in name:
                module.register_forward_hook(
                    lambda *args, **kwargs: HookSimpleMLPLayer.get_forward(self, *args, **kwargs))
                module.register_full_backward_hook(
                    lambda *args, **kwargs: HookSimpleMLPLayer.get_backward(self, *args, **kwargs))
                    
    def get_forward(self, module, input, output):
        self.layer_forward.append(output.cpu())

    def get_backward(self, module, grad_input, grad_output):
        self.layer_backward.append(grad_output[0].cpu())
        
    def get_forward_mean(self):
        # returns the largest activactions on average
        return torch.mean(torch.stack(self.layer_forward), dim=0)

    def get_backward_mean(self):
        # returns the largest activactions on average
        return torch.mean(torch.stack(self.layer_backward), dim=0)

class HookSimpleMLP:
    """ Simple model that hooks all layers in the simple model.
        Given a single input, it keeps track of the forward and backward passes 
        of each layer. All outputs will be appended one after the other.
        NOTE: requires a loss function that is being used for computing the gradients
    """
    def __init__(self, model, loss_fn):
        self.model = copy.deepcopy(model)
        self.loss_fn = loss_fn
        self.layers = {}
        self.device = "cpu"#torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
        self.model.to(self.device)
        # Register hooks at all layers
        for name, module in self.model.named_modules():
            if name.strip() == "":
                # Ignore the whole module name
                continue
            if "act" in name:
                # We ignore activation functions!
                continue
            self.layers[name] = HookSimpleMLPLayer(self.model, name)

    def __call__(self, input, label):
        self.model.zero_grad()
        output = self.model(input.to(self.device))
        loss = self.loss_fn(output, label)
        loss.backward()
        
    def set_instances(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def run_instances(self):
        try:
            for x, y in zip(self.x_data, self.y_data):
                self.__call__(x, y)
        except AttributeError:
            logging.error("No data found. Please first set a dataset using set_instances(x_data, y_data) .")
            
    def get_largest_forward_mean(self):
        # Identifies the neurons with the largest activation values (on average) per layer
        return [(k, torch.argmax(v).item()) for k,v in self.get_forward_mean()]

    def get_largest_backward_mean(self):
        # Identifies the neurons with the largest activation values (on average) per layer
        return [(k, torch.argmax(v).item()) for k,v in self.get_backward_mean()]        

    def get_forward_mean(self):
        # Identifies the neurons with the largest activation values (on average) per layer
        result = []
        for k, v in self.layers.items():
            result.append((k, v.get_forward_mean()))
        return result
        
    def get_backward_mean(self):
        # Identifies the neurons with the largest activation values (on average) per layer
        result = []
        for k, v in self.layers.items():
            result.append((k, v.get_backward_mean()))
        return result     
        
    
