import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class FaultDataset(Dataset):
    """Fault dataset."""

    def __init__(self, data: list):
        """
        Arguments:
            data: a list of faults consisting of 
                observation - the state (input)
                faulty_action - the bad action
                action_mask - all valid actions
                action (optional) - an action that does not lead to a fault
        """
        self.data = data
        # output dimension (important for one hot encoding)
        self.action_dim = len(data[0]["action_mask"])
        self.preprocess_data()
        
    def preprocess_data(self):
        # create lists of fault_action-instance and (valid) random_action-instance 
        self.inputs = []
        self.faulty_actions = []
        self.all_valid_actions = []
        self.sampled_valid_actions = []
        for element in self.data:
            self.inputs.append(element["observation"])
            # get indices of valid actions:
            # self.faulty_actions.append(F.one_hot(torch.tensor(element["faulty_action"]), self.action_dim).to(torch.float))
            self.faulty_actions.append(torch.tensor(element["faulty_action"]))
            # all valid actions
            all_actions = element["action_mask"]
            # mask faulty action
            all_actions[element["faulty_action"]] = 0.
            # indices of all valid actions that are not a fault
            all_actions_idx = torch.flatten(torch.nonzero(torch.tensor(all_actions)))
            self.all_valid_actions.append(all_actions_idx)
            valid_idx = random.randint(0, len(all_actions_idx)-1)
            #self.sampled_valid_actions.append(F.one_hot(all_actions_idx[valid_idx], self.action_dim).to(torch.float))
            self.sampled_valid_actions.append(all_actions_idx[valid_idx])
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        
        return {"input":self.inputs[idx], "fault":self.faulty_actions[idx], "valid":self.sampled_valid_actions[idx], "all_valid":self.all_valid_actions[idx]}
