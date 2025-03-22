import torch
import torch.nn.functional as F

def pad_tensors_to_max_size(tensor_list):
    # Find maximum dimensions
    max_dims = []
    for dim in range(tensor_list[0].dim()):
        max_dim_size = max([tensor.size(dim) for tensor in tensor_list])
        max_dims.append(max_dim_size)
    
    # Pad each tensor to match max dimensions
    padded_tensors = []
    for tensor in tensor_list:
        # Calculate padding for each dimension
        pad_sizes = []
        for dim in range(tensor.dim()):
            pad_size = max_dims[dim] - tensor.size(dim)
            # Padding is applied from the last dimension backward
            # For each dimension, we need (padding_left, padding_right)
            # We'll add all padding to the right side
            pad_sizes = [0, pad_size] + pad_sizes
        
        # Apply padding
        padded_tensor = F.pad(tensor, pad_sizes, mode='constant', value=0)
        padded_tensors.append(padded_tensor)
    
    return torch.stack(padded_tensors)

def collate_auctions(batch):
    inputs, targets = zip(*batch)
    auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values = zip(*inputs)

    auctions = pad_tensors_to_max_size(auctions)
    item_index = pad_tensors_to_max_size(item_index)
    contexts = pad_tensors_to_max_size(contexts)
    bonus_lists = pad_tensors_to_max_size(bonus_lists)
    modifier_types = pad_tensors_to_max_size(modifier_types)
    modifier_values = pad_tensors_to_max_size(modifier_values)
    targets = pad_tensors_to_max_size(targets)

    return (auctions, item_index, contexts, bonus_lists, modifier_types, modifier_values), targets
