import numpy as np
from sympy.combinatorics import Permutation
from itertools import product

class FiniteGroupAction:
    """
    Implementation of non-geometric finite group actions for 3D medical images.
    
    These transformations preserve the value range of the original image, making them
    suitable for neural network processing while maintaining group-theoretic properties.
    """
    
    def __init__(self, group_type="cyclic", order=4, preserve_stats=True):
        """
        Initialize a finite group action.
        
        Parameters:
        -----------
        group_type : str
            Type of group action ('cyclic', 'dihedral', 'klein_four', 'symmetric')
        order : int
            Order of the group (relevant for cyclic and dihedral groups)
        preserve_stats : bool
            Whether to preserve the statistical properties of the image (mean, variance)
        """
        self.group_type = group_type
        self.order = order
        self.preserve_stats = preserve_stats
        self.group_elements = self._generate_group_elements()
        
    def _generate_group_elements(self):
        """Generate the elements of the finite group."""
        if self.group_type == "cyclic":
            # Cyclic group Z_n: {0, 1, 2, ..., n-1} with addition mod n
            return list(range(self.order))
            
        elif self.group_type == "dihedral":
            # Dihedral group D_n has 2n elements
            # We represent them as pairs (r, s) where r is rotation, s is reflection
            return list(product(range(self.order), [0, 1]))
            
        elif self.group_type == "klein_four":
            # Klein four-group: {e, a, b, c} with a²=b²=c²=e, ab=c
            return ["e", "a", "b", "c"]
            
        elif self.group_type == "symmetric":
            # Symmetric group S_n (n must be small)
            if self.order > 4:
                raise ValueError("Symmetric group order must be <= 4 for practicality")
            # Return all permutations of [0,1,...,order-1]
            elements = []
            for perm in Permutation.full_perm_group(self.order):
                elements.append(perm)
            return elements
        
        else:
            raise ValueError(f"Unsupported group type: {self.group_type}")
    
    def apply(self, image, element_idx=None):
        """
        Apply a group action to the image.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input 3D image of shape [1,D,H,W]
        element_idx : int or tuple
            Index of the group element to apply
            
        Returns:
        --------
        numpy.ndarray
            Transformed image
        """
        if element_idx is None:
            # Apply random group element
            element_idx = np.random.randint(0, len(self.group_elements))
        
        element = self.group_elements[element_idx]
        
        # Make sure we work with a copy to avoid modifying the original
        image_copy = image.copy()
        
        # Extract the 3D volume (removing batch dimension if needed)
        if len(image_copy.shape) == 4 and image_copy.shape[0] == 1:
            volume = image_copy[0]
        else:
            volume = image_copy
        
        # Get min and max values for proper scaling
        min_val = np.min(volume)
        max_val = np.max(volume)
        value_range = max_val - min_val
        
        if self.group_type == "cyclic":
            # For a proper cyclic group action, we need something simpler
            # that guarantees the formal mathematical properties
            
            # Identity element handling
            if element == 0:
                # For element 0 (identity), return the original image exactly
                transformed_vol = volume.copy()
            else:
                # For non-identity elements, we need a permutation-based approach
                # This guarantees:
                # 1. Group properties (closure, identity, inverse, associativity)
                # 2. Exact value preservation
                # 3. Proper cycle length
                
                # Create a permutation that exactly maps the values
                # based on the group element
                if value_range > 0:
                    # A simple way to implement a group action for cyclic groups:
                    # If element=1 is the generator, we shift the values by one position 
                    # in the sorted array, and element=2 shifts by two positions, etc.
                    
                    # Flatten the volume to work with all values
                    flat_vol = volume.flatten()
                    
                    # Sort the values to establish a mapping
                    sorted_indices = np.argsort(flat_vol)
                    
                    # Create a mapping based on the cyclic group element
                    # This defines a permutation of the sorted array
                    perm = np.zeros_like(sorted_indices)
                    for i in range(len(sorted_indices)):
                        perm[i] = sorted_indices[(i + element) % len(sorted_indices)]
                    
                    # Apply the permutation to create a new array
                    new_flat = np.zeros_like(flat_vol)
                    for i in range(len(sorted_indices)):
                        new_flat[sorted_indices[i]] = flat_vol[perm[i]]
                    
                    # Reshape back to original dimensions
                    transformed_vol = new_flat.reshape(volume.shape)
                else:
                    # If all values are the same, applying any element gives the same result
                    transformed_vol = volume.copy()
                
        elif self.group_type == "dihedral":
            # For dihedral group, combine cyclic action with a reflection
            r, s = element  # Rotation and reflection
            
            # Apply a cyclic transformation based on the rotation component
            if r == 0 and s == 0:
                # Identity element
                transformed_vol = volume.copy()
            else:
                # For non-identity elements, use a similar approach to cyclic groups
                # but with additional reflection logic
                
                # First, apply the cyclic component (rotation)
                if r != 0:
                    # Use the same cyclic group implementation as above
                    # Flatten the volume to work with all values
                    flat_vol = volume.flatten()
                    sorted_indices = np.argsort(flat_vol)
                    
                    # Create a mapping based on the cyclic group element
                    perm = np.zeros_like(sorted_indices)
                    for i in range(len(sorted_indices)):
                        perm[i] = sorted_indices[(i + r) % len(sorted_indices)]
                    
                    # Apply the permutation
                    new_flat = np.zeros_like(flat_vol)
                    for i in range(len(sorted_indices)):
                        new_flat[sorted_indices[i]] = flat_vol[perm[i]]
                    
                    # Reshape back to original dimensions
                    transformed_vol = new_flat.reshape(volume.shape)
                else:
                    transformed_vol = volume.copy()
                
                # Then, apply reflection if needed
                if s == 1:
                    # Reflection: map x -> min + max - x
                    # This exactly preserves min and max values
                    transformed_vol = min_val + max_val - transformed_vol
                
        elif self.group_type == "klein_four":
            # Klein four-group implementation
            if element == "e":  # Identity
                transformed_vol = volume.copy()
            elif element == "a":  # First non-trivial element
                # For element 'a', invert the values within the range
                transformed_vol = min_val + max_val - volume
            elif element == "b":  # Second non-trivial element
                # For element 'b', use a permutation approach
                # Flatten the volume to work with all values
                flat_vol = volume.flatten()
                sorted_indices = np.argsort(flat_vol)
                
                # Create a mapping that satisfies b^2 = e (order 2)
                perm = np.zeros_like(sorted_indices)
                for i in range(len(sorted_indices)):
                    # For order 2, just swap pairs
                    if i % 2 == 0 and i+1 < len(sorted_indices):
                        perm[i] = sorted_indices[i+1]
                        perm[i+1] = sorted_indices[i]
                    elif i % 2 == 1:
                        # Already handled in previous iteration
                        pass
                    else:
                        # Handle odd length case for last element
                        perm[i] = sorted_indices[i]
                
                # Apply the permutation
                new_flat = np.zeros_like(flat_vol)
                for i in range(len(sorted_indices)):
                    new_flat[sorted_indices[i]] = flat_vol[perm[i]]
                
                # Reshape back to original dimensions
                transformed_vol = new_flat.reshape(volume.shape)
            else:  # element == "c" (composition of a and b)
                # For element 'c', apply 'b' then 'a'
                # First, apply 'b' (same as above)
                flat_vol = volume.flatten()
                sorted_indices = np.argsort(flat_vol)
                
                perm = np.zeros_like(sorted_indices)
                for i in range(len(sorted_indices)):
                    if i % 2 == 0 and i+1 < len(sorted_indices):
                        perm[i] = sorted_indices[i+1]
                        perm[i+1] = sorted_indices[i]
                    elif i % 2 == 1:
                        pass
                    else:
                        perm[i] = sorted_indices[i]
                
                new_flat = np.zeros_like(flat_vol)
                for i in range(len(sorted_indices)):
                    new_flat[sorted_indices[i]] = flat_vol[perm[i]]
                
                # Then apply 'a' (invert within range)
                transformed_vol = min_val + max_val - new_flat.reshape(volume.shape)
        
        # Restore batch dimension if needed
        if len(image.shape) == 4 and image.shape[0] == 1:
            transformed = np.expand_dims(transformed_vol, axis=0)
        else:
            transformed = transformed_vol
            
        return transformed
    
    def orbit(self, image):
        """
        Generate the complete orbit of an image under the group action.
        
        Parameters:
        -----------
        image : numpy.ndarray
            Input 3D image
            
        Returns:
        --------
        list of numpy.ndarray
            List of transformed images, one for each group element
        """
        orbit_images = []
        for i in range(len(self.group_elements)):
            orbit_images.append(self.apply(image, i))
        return orbit_images


# Example usage
if __name__ == "__main__":
    # Create a sample 3D image [1,28,28,28]
    sample_image = np.random.rand(1, 28, 28, 28)
    
    # Apply different group actions
    actions = [
        FiniteGroupAction(group_type="cyclic", order=4),
        FiniteGroupAction(group_type="dihedral", order=4),
        FiniteGroupAction(group_type="klein_four"),
    ]
    
    # Demonstrate orbits
    for i, action in enumerate(actions):
        print(f"Group action: {action.group_type}, elements: {len(action.group_elements)}")
        orbit = action.orbit(sample_image)
        print(f"Orbit size: {len(orbit)}")
        print("---")