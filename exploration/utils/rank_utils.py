import torch

def effective_dimension(matrix: torch.tensor, method='singular') -> torch.tensor:
    """
    Computes the effective entropy of a matrix.
    Based on https://infoscience.epfl.ch/server/api/core/bitstreams/2907ab8a-23f5-481d-bb07-1d56a3f3511f/content
    
    Parameters:
    -----------
    matrix : torch.Tensor
        Input matrix for entropy computation
    method : str, optional
        Method to use for entropy computation:
        - 'singular': Based on normalized singular values (default)
    Returns:
    --------
    float
        The effective entropy value
    
    Notes:
    ------
    For singular value entropy: H = -sum(p_i * log(p_i)) where p_i = s_i/sum(s_i)
    """
    if matrix.ndimension() < 2:
        raise ValueError("Input must be at least a 2D matrix")
    
    if method == 'singular':
        # Compute singular values
        s = torch.linalg.svdvals(matrix)
        # Normalize singular values to get "probability" distribution
        p = s / torch.sum(s.abs())
        # Remove zeros to avoid log(0)
        p = p[p > 0]
        # Compute entropy
        H = -torch.sum(p * torch.log(p))
        return torch.exp(H)
    else:
        raise ValueError("Invalid method")
    

if __name__ == "__main__":
    # Test effective dimension
    matrix = torch.randn(120, 120)
    print(f"Effective dimension of random matrix: {effective_dimension(matrix)}")
    matrix = torch.eye(120)
    print(f"Effective dimension of identity matrix: {effective_dimension(matrix)}")
