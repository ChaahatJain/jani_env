import logging 

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

import torch
from torch import nn
        
from tracing.hooked_model import HookSimpleMLP
from tracing.manipulated_model import ManipulatedSimpleMLP
from nn.simple_model import SimpleMLP

def compute_diff_abs(layers1, layers2):
    # Computes the absolute distance between two lists of layer-wise means
    distances = []
    for (name1, vec1), (name2, vec2) in zip(layers1, layers2):
        assert(name1 == name2)
        distances.append((name1, torch.abs(torch.diff(torch.stack([vec1, vec2]), dim=0))))
    return distances
    
def get_max_neurons(layers):
    # Gets a list of mmean vectors and returns the respective max neurons
    return [(k, torch.argmax(v).item()) for k,v in layers]
    
def sum_layers(layers1, layers2):
    summed = []
    for (name1, vec1), (name2, vec2) in zip(layers1, layers2):
        assert(name1 == name2)
        summed.append((name1, torch.sum(torch.stack([vec1, vec2]), 0)))
    return summed

def trace_max_diff_forward(model, loss_fn, trace, remain):
    """ Traces a single instance (x,y) pair given a specific dataset ([x],[y]) pair
        Returns a list of a single neuron per layer where:
        max( abs(diff(trace, pos)) + abs(diff(trace,neg)) )
        I.e., the neuron where the difference to the mean positive and negative sets is largest  
        Note: dataset should not contain the traced instance.
    """
    x_trace, y_trace = trace
    x_remain, y_remain = remain
    sanity_check_weights = model.hidden1.weight.clone()
    
    # Trace positive, negative, and the tracing instance separately
    hooked_positive = HookSimpleMLP(model, loss_fn)
    hooked_negative = HookSimpleMLP(model, loss_fn)
    hooked_trace = HookSimpleMLP(model, loss_fn)

    # Split the remaining data into positive examples (containing traced action)
    # and negative examples (containing any other action)
    x_positive = [x for idx, x in enumerate(x_remain) if torch.argmax(y_remain[idx]) == torch.argmax(y_trace)]
    y_positive = [y for idx, y in enumerate(y_remain) if torch.argmax(y_remain[idx]) == torch.argmax(y_trace)]

    x_negative = [x for idx, x in enumerate(x_remain) if torch.argmax(y_remain[idx]) != torch.argmax(y_trace)]
    y_negative = [y for idx, y in enumerate(y_remain) if torch.argmax(y_remain[idx]) != torch.argmax(y_trace)]

    # Trace the respective instance
    # You can trace individual instances by directly calling the respective class with the instances
    hooked_trace(x_trace, y_trace)

    # You can trace multiple instances by setting them first and then calling run_instances()
    hooked_negative.set_instances(x_negative, y_negative)
    hooked_negative.run_instances()

    hooked_positive.set_instances(x_positive, y_positive)
    hooked_positive.run_instances()
        
    # Make sure model weights stay constant!
    if not torch.equal(sanity_check_weights, model.hidden1.weight):
        logging.error(" You are updating the weights during tracing! This will falsify your results")  

    change_positive = hooked_positive.get_forward_mean()
    change_negative = hooked_negative.get_forward_mean()
    change_trace = hooked_trace.get_forward_mean() 

    trace_pos = compute_diff_abs(change_trace, change_positive)
    trace_neg = compute_diff_abs(change_trace, change_negative)
    
    trace_sum = sum_layers(trace_pos, trace_neg)
    
    return get_max_neurons(trace_sum)
    
def trace_max_diff_backward(model, loss_fn, trace, remain):
    """ Traces a single instance (x,y) pair given a specific dataset ([x],[y]) pair
        Returns a list of a single neuron per layer where:
        max( abs(diff(trace, pos)) + abs(diff(trace,neg)) )
        I.e., the neuron where the difference to the mean positive and negative sets is largest  
        Note: dataset should not contain the traced instance.
    """
    x_trace, y_trace = trace
    x_remain, y_remain = remain
    sanity_check_weights = model.hidden1.weight.clone()
    
    # Trace positive, negative, and the tracing instance separately
    hooked_positive = HookSimpleMLP(model, loss_fn)
    hooked_negative = HookSimpleMLP(model, loss_fn)
    hooked_trace = HookSimpleMLP(model, loss_fn)

    # Split the remaining data into positive examples (containing traced action)
    # and negative examples (containing any other action)
    x_positive = [x for idx, x in enumerate(x_remain) if torch.argmax(y_remain[idx]) == torch.argmax(y_trace)]
    y_positive = [y for idx, y in enumerate(y_remain) if torch.argmax(y_remain[idx]) == torch.argmax(y_trace)]

    x_negative = [x for idx, x in enumerate(x_remain) if torch.argmax(y_remain[idx]) != torch.argmax(y_trace)]
    y_negative = [y for idx, y in enumerate(y_remain) if torch.argmax(y_remain[idx]) != torch.argmax(y_trace)]

    # Trace the respective instance
    # You can trace individual instances by directly calling the respective class with the instances
    hooked_trace(x_trace, y_trace)

    # You can trace multiple instances by setting them first and then calling run_instances()
    hooked_negative.set_instances(x_negative, y_negative)
    hooked_negative.run_instances()

    hooked_positive.set_instances(x_positive, y_positive)
    hooked_positive.run_instances()
        
    # Make sure model weights stay constant!
    if not torch.equal(sanity_check_weights, model.hidden1.weight):
        logging.error(" You are updating the weights during tracing! This will falsify your results")  

    change_positive = hooked_positive.get_backward_mean()
    change_negative = hooked_negative.get_backward_mean()
    change_trace = hooked_trace.get_backward_mean() 

    trace_pos = compute_diff_abs(change_trace, change_positive)
    trace_neg = compute_diff_abs(change_trace, change_negative)
    
    trace_sum = sum_layers(trace_pos, trace_neg)
    
    return get_max_neurons(trace_sum)
    
def trace_max_forward(model, loss_fn, trace):
    """ Traces a single instance (x,y)
        Returns a list of a single neuron per layer where:
        max( trace )
        I.e., the neuron with the largest forward activation
    """
    x_trace, y_trace = trace
    sanity_check_weights = model.hidden1.weight.clone()
    hooked_trace = HookSimpleMLP(model, loss_fn)
    hooked_trace(x_trace, y_trace)
    # Make sure model weights stay constant!
    if not torch.equal(sanity_check_weights, model.hidden1.weight):
        logging.error(" You are updating the weights during tracing! This will falsify your results")  

    change_trace = hooked_trace.get_forward_mean() 
   
    return get_max_neurons(change_trace)
    
def trace_max_backward(model, loss_fn, trace):
    """ Traces a single instance (x,y)
        Returns a list of a single neuron per layer where:
        max( trace )
        I.e., the neuron with the largest forward activation
    """
    x_trace, y_trace = trace
    sanity_check_weights = model.hidden1.weight.clone()
    hooked_trace = HookSimpleMLP(model, loss_fn)
    hooked_trace(x_trace, y_trace)
    # Make sure model weights stay constant!
    if not torch.equal(sanity_check_weights, model.hidden1.weight):
        logging.error(" You are updating the weights during tracing! This will falsify your results")  

    change_trace = hooked_trace.get_backward_mean() 
   
    return get_max_neurons(change_trace)


#def evaluate_changes(model, loss_fn, trace, remain):
    


