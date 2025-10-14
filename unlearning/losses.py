import torch
import torch.nn as nn
import torch.nn.functional as F

'''
These functions are used to compute the loss for the unlearning process.
The loss functions are defined in the config file.

We expect the loss to be described in the format forget-loss_retain-loss (eg: GA_GD)
The loss is then computed in the get_loss function. 
You can easily add new loss functions by adding a individual loss function and updating the get_loss function.
'''

########## Main Loss Function ##########

def get_simple_loss(model, input_forget, input_retain):
    '''
    This function computes the loss for the unlearning process.
    '''
    # forget_loss
    forget_loss = ga_simple_loss(model, input_forget)

    # regularization_loss
    regularization_loss = gd_simple_loss(model, input_retain)

    return forget_loss, regularization_loss

########## Individual Loss Functions ##########

# Forget Loss: GA
def ga_simple_loss(model, input_forget):
    # The first element of the data tuple is the target data
    x_forget, y_forget = input_forget
    # Compute the Cross entropy loss for the answer
    loss_fn = torch.nn.CrossEntropyLoss() 
    y_pred = model(x_forget)
    #reversing the sign for gradient ascent
    forget_loss = -1 * loss_fn(y_forget, y_pred)
    return  forget_loss

# Regularization Loss: GD
def gd_simple_loss(model, input_retain):
    x_retain, y_retain = input_retain
    # Compute the Cross entropy loss for the answer
    loss_fn = torch.nn.CrossEntropyLoss() 
    y_pred = model(x_retain)   
    retain_loss = loss_fn(y_retain, y_pred)

    return retain_loss

