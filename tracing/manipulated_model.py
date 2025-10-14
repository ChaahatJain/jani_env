import copy
import random
import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import numpy as np
import torch
from torch import nn
        
class ManipulatedSimpleMLP:
    """ Simple model with simple manipulation operations
    """
    def __init__(self, model):       
        self.model = model
        self.checking_layer = model.hidden1.weight.clone()
        self.change_val = 0.
        self.change_func = "value"

    def set_instances(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data
        
    def get_manipulated_model(self):
        try:
            return self.new_model
        except AttributeError:
            logging.error(" No manipulated model found. Please modify the model first!")


    def run_instances(self):
        try:
            outputs = []
            for x, y in zip(self.x_data, self.y_data):
                self.model.zero_grad()
                outputs.append(self.model(x))
            return outputs
        except AttributeError:
            logging.error(" No data found. Please first set a dataset using set_instances(x_data, y_data) .")

    def update_weight(self, input=None):
        if self.change_func == "value":
            return self.change_val
        elif self.change_func == "increase-value":
            return input + self.change_val
        elif self.change_func == "decrease-value":
            return input - self.change_val
        elif self.change_func == "random-uniform":
            return np.random.uniform()
        elif self.change_func == "increase-uniform":
            return input + np.random.uniform()
        elif self.change_func == "decrease-uniform":
            return input - np.random.uniform()
        
    def set_change_func(self, func_name):
        self.change_func = func_name
            
    def set_change_val(self, change_val):
        self.change_val = change_val

    def run_instances_manipulated(self):
        try:
            self.new_model
        except AttributeError:
            logging.error(" No manipulated model found. Please create one first.")
            return None
        try:
            outputs = []
            # Check that the model weights have actually changed!
            assert(not torch.equal(self.new_model.hidden1.weight, self.checking_layer))
            for x, y in zip(self.x_data, self.y_data):
                self.new_model.zero_grad()
                outputs.append(self.new_model(x))
            return outputs
        except AttributeError:
            logging.error(" No data found. Please first set a dataset using set_instances(x_data, y_data) .")


    def run_instances_random(self):
        try:
            self.random_model
        except AttributeError:
            logging.info(" No random model found. Initializing...")
            self.generate_random_model()
        try:
            outputs = []
            # Check that the model weights have actually changed!
            assert(not torch.equal(self.random_model.hidden1.weight, self.checking_layer))
            for x, y in zip(self.x_data, self.y_data):
                self.random_model.zero_grad()
                outputs.append(self.random_model(x))
            return outputs
        except AttributeError:
            logging.error(" No data found. Please first set a dataset using set_instances(x_data, y_data) .")


    def generate_random_model(self):
        self.random_model = copy.deepcopy(self.model)
        with torch.no_grad():
            for name, rand_layer in self.random_model.named_modules():
                if name == "output" or name.strip() == "" or "act" in name:
                    # Skip any activation functions and SimpleMLP modules
                    continue 
                rand_idx = random.randint(0, len(rand_layer.bias)-1)
                rand_layer.weight = nn.Parameter(rand_layer.weight)
                rand_layer.bias = nn.Parameter(rand_layer.bias)
                rand_layer.weight[rand_idx] = self.update_weight(rand_layer.weight[rand_idx])
                rand_layer.bias[rand_idx] = self.update_weight(rand_layer.bias[rand_idx])
                logging.info(f" Randomly turning off neuron {rand_idx} in layer {name}.")
            

    def manipulate_model_params(self, changes):
        self.new_model = copy.deepcopy(self.model)
        with torch.no_grad():
            for name, neuron_idx in changes:
                if name == "output":
                    # Skip softmax layer
                    continue 
                for new_name, layer in self.new_model.named_modules():
                    if name == new_name:
                        layer.weight = nn.Parameter(layer.weight)
                        layer.bias = nn.Parameter(layer.bias)
                        layer.weight[neuron_idx] = self.update_weight(layer.weight[neuron_idx])
                        layer.bias[neuron_idx] = self.update_weight(layer.bias[neuron_idx])
                        logging.info(f" Turning off neuron {neuron_idx} in layer {name}.")
