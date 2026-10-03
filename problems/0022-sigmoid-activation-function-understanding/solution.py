import torch

def sigmoid(z: float) -> float:
    """
    Compute the sigmoid activation function.
    Input:
      - z: float or torch scalar tensor
    Returns:
      - sigmoid(z) as Python float rounded to 4 decimals.
    """
    z = torch.tensor(z) if not isinstance(z, torch.Tensor) else z
    return round(torch.sigmoid(z).item(),4)
    pass
