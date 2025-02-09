#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from itertools import product

from . import pd, np, BaseEstimator


# In[ ]:


class PolynomialBasis(BaseEstimator):
    def __init__(self, max_order: int = 3):
        """
        Initialize the PolynomialBasis estimator with a maximum polynomial order.
        
        Parameters:
        max_order (int): The maximum order of polynomials to include for each feature, excluding the constant term.
        """
        self.max_order = max_order
        
    def __call__(self, data: np.ndarray):
        """
        Compute polynomial basis functions for each feature independently, with a single constant term (x^0).
        
        Parameters:
        data (np.ndarray): Input data with shape (n_samples, n_features).
        
        Returns:
        tuple: A DataFrame of basis functions and a dictionary mapping column names to polynomial orders for each sample.
        """
        n_samples, n_features = data.shape
        basis_values = []
        basis_dict = {}

        # Add the constant term (x^0) to the basis functions (this term is the same for all samples)
        constant_term = np.ones(n_samples)
        basis_values.append(constant_term)
        basis_dict["Constant"] = (0, 0)  # No feature associated with constant term, order 0

        # Iterate over each feature and compute its polynomial terms (from x^1 to x^max_order)
        for feature_index in range(n_features):
            feature_column = data[:, feature_index]
            for order in range(1, self.max_order + 1):  # Start from order 1, as order 0 is already added
                # Compute the polynomial term for the current feature and order
                poly_term = feature_column ** order
                column_name = f'Feature_{feature_index+1}_Order_{order}'
                basis_dict[column_name] = (feature_index+1, order)
                basis_values.append(poly_term)

        # Create DataFrame from basis values
        basis_df = pd.DataFrame(np.column_stack(basis_values))
        basis_df.columns = basis_dict.keys()
        
        return basis_df, basis_dict


# In[ ]:


class BasisNextExpect(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression to calculate the next expectation
    of basis functions conditioned on state variables.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel, affecting the
        smoothness of the kernel density estimation.
    """

    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with specified bandwidth and regularization alpha
        self.bandwidth = bandwidth

    def fit(self, X, X_next, basis):
        """
        Fit the model by storing the current and next state data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The current state data.
        X_next : array-like, shape (n_samples, n_features)
            The next state data.
            
        basis : object
            An object representing the basis function model, providing a method to compute basis 
            functions based on the given orders and input data.
        
        Returns
        -------
        self : object
            Returns self with stored data.
        """
        self.X_ = X # Store the current state data in the instance
        self.X_next = X_next # Store the next state data in the instance
        
        # Compute the basis functions for the concatenated next state data, using the basis function model.
        # basis_next_df contains the evaluated basis functions for each extended next state.
        self.basis_next_df, _ = basis(self.X_next)
        
        return self
    
    def __call__(self, data_matrix):
        """
        Apply the basis_next_expect function to each row of the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.
        
        Returns
        -------
        BNE_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the basis_next_expect method to each row in data_matrix
        BNE_vec = np.apply_along_axis(self.basis_next_expect, 1, data_matrix, self.bandwidth)
        return BNE_vec
    
    def basis_next_expect(self, x, h_x):
        """
        Calculate the conditional expectation of basis functions for the next state, 
        conditioned on the current state, using Kernel Density Estimation (KDE).

        Parameters
        ----------
        x : array-like, shape (p,)
            The conditioning variable (current state) for which the conditional expectation is computed.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) controlling the smoothness of the Gaussian kernel for each dimension.

        Returns
        -------
        BNE : ndarray
            The conditional expectation value(s) of the basis functions for the next state, given the current state `x`.
        """   
        # Calculate the pairwise differences between `x` (current state) and the fitted states `self.X_`
        u_x = (x - self.X_) / h_x # Normalized difference between the input and stored states
        
        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor

        # Compute the numerator as the weighted sum of basis function values for the next state.
        # Perform element-wise multiplication of basis function values with kernel weights.
        BNE_num = np.sum(self.basis_next_df.values * K_x[:, np.newaxis], axis=0)

        # Calculate the denominator as the sum of kernel weights.
        # Add a small constant `1e-10` to prevent division by zero and ensure numerical stability.
        BNE_denom = np.sum(K_x) + 1e-10
        
        # Compute the conditional expectation by dividing the weighted sum of basis functions by the total kernel weights.
        BNE = BNE_num/BNE_denom
        
        return BNE


# In[ ]:


class BasisNextSAExpect(BaseEstimator):
    """
    Custom estimator for calculating the conditional expectation of the next state 
    and action values based on Nadaraya-Watson kernel regression and basis functions.
    This estimator is used to compute the expected values of basis functions given the 
    current state-action pair and leverages a Gaussian kernel for smoothing.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel, 
        affecting the smoothness of the kernel density estimation.
    alpha : float, default=0.1
        The regularization term added to the kernel weights to ensure numerical stability.
        Adding a small positive value helps prevent extremely small or zero weights, which
        could otherwise result in unstable or biased estimates.
    """
    
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        """
        Initialize the BasisNextSAExpect estimator with specified parameters.

        Parameters
        ----------
        bandwidth : float, default=1.0
            The bandwidth parameter for the Gaussian kernel, affecting the influence 
            range of each data point in the kernel density estimation.
        alpha : float, default=0.1
            Regularization term added to the kernel weights to avoid issues such as 
            division by zero during kernel-based computations.
        """
        self.bandwidth = bandwidth
        self.alpha = alpha

    def fit(self, X, A, X_next, basis):
        """
        Fit the model by storing the current state, action, and next state data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The current state data.
        A : array-like, shape (n_samples,)
            The action labels corresponding to each sample in X.
        X_next : array-like, shape (n_samples, n_features)
            The next state data.
            
        basis : object
            An object representing the basis function model, providing a method to compute basis 
            functions based on the given orders and input data.

        Returns
        -------
        self : object
            Returns the fitted estimator with stored data.
        """
        self.X_ = X  # Store current state data in the instance
        self.A_ = A  # Store action labels in the instance
        self.X_next = X_next  # Store next state data in the instance
        
        # Compute the basis functions for the concatenated next state data, using the basis function model.
        # basis_next_df contains the evaluated basis functions for each extended next state.
        self.basis_next_df, _ = basis(self.X_next)
        
        return self

    def __call__(self, data_matrix, a):
        """
        Apply the basis_next_expect function to each row of the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the conditional expectation.
        a : scalar
            The action label to use for conditioning.
        basis : object
            The basis object for computing basis functions.

        Returns
        -------
        BNE_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the basis_next_expect function to each row in data_matrix
        BNE_vec = np.apply_along_axis(self.basis_next_expect, 1, data_matrix, a, self.bandwidth)
        return BNE_vec
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.

        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare.

        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements are True if action 'a' matches
        ind = vec == a
        return ind

    def basis_next_expect(self, x, a, h_x):
        """
        Calculate the conditional expectation of the basis functions for the next state
        given the current state-action pair, using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : array-like, shape (p,)
            The current state (conditioning variable).
        a : scalar
            The action label to condition the expectation on.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) for the Gaussian kernel.
        basis : object
            The basis object used to compute basis functions, providing the basis functions 
            and their computation method.

        Returns
        -------
        BNE : ndarray
            The estimated conditional expectation value(s) for the given state-action pair.
        """
        
        # Compute the normalized pairwise distances between the input state-action pair and the stored training data
        u_x = (x - self.X_) / h_x # Scale differences for states by bandwidth
        
        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # Binary indicator array for action `a`
        
        # Multiply the Gaussian kernel weights by the action indicator for joint state-action smoothing
        K= K_x * ind_A
        #K+=self.alpha # Adding self.alpha ensures regularization to prevent overfitting

        # Compute the numerator: weighted sum of the basis function values for the next state
        BNE_num = np.sum(self.basis_next_df.values * K[:, np.newaxis], axis=0) # Element-wise multiplication of kernel weights
        
        # Calculate the denominator as the sum of kernel weights
        # A small constant (1e-10) is included to avoid division by zero
        BNE_denom = np.sum(K) + 1e-10
        
        # Calculate the conditional expectation by dividing the weighted sum of basis functions by the kernel weights
        BNE = BNE_num/BNE_denom
        
                      
            
        return BNE


# In[ ]:


class BasisNextSAExpectData(BaseEstimator):
    """
    Custom estimator for calculating the conditional expectation of the next state 
    and action values based on Nadaraya-Watson kernel regression and basis functions.
    This estimator is used to compute the expected values of basis functions given the 
    current state-action pair and leverages a Gaussian kernel for smoothing.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter that controls the width of the Gaussian kernel, 
        affecting the smoothness of the kernel density estimation.
    alpha : float, default=0.1
        The regularization term added to the kernel weights to ensure numerical stability.
        Adding a small positive value helps prevent extremely small or zero weights, which
        could otherwise result in unstable or biased estimates.
    """
    
    def __init__(self, bandwidth=1.0, alpha = 0.1):
        """
        Initialize the BasisNextSAExpect estimator with specified parameters.

        Parameters
        ----------
        bandwidth : float, default=1.0
            The bandwidth parameter for the Gaussian kernel, affecting the influence 
            range of each data point in the kernel density estimation.
        alpha : float, default=0.1
            Regularization term added to the kernel weights to avoid issues such as 
            division by zero during kernel-based computations.
        """
        self.bandwidth = bandwidth
        self.alpha = alpha

    def fit(self, X, A, X_next, basis):
        """
        Fit the model by storing the current state, action, and next state data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The current state data.
        A : array-like, shape (n_samples,)
            The action labels corresponding to each sample in X.
        X_next : array-like, shape (n_samples, n_features)
            The next state data.
            
        basis : object
            An object representing the basis function model, providing a method to compute basis 
            functions based on the given orders and input data.

        Returns
        -------
        self : object
            Returns the fitted estimator with stored data.
        """
        self.X_ = X  # Store current state data in the instance
        self.A_ = A  # Store action labels in the instance
        self.X_next = X_next  # Store next state data in the instance
        
        # Compute the basis functions for the concatenated next state data, using the basis function model.
        # basis_next_df contains the evaluated basis functions for each extended next state.
        self.basis_next_df, _ = basis(self.X_next)
        
        return self

    def __call__(self, data_matrix, a_vec):
        """
        Apply the basis_next_expect function to each row of the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the conditional expectation.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.
        basis : object
            The basis object for computing basis functions.

        Returns
        -------
        BNE_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix, for action 'a'.
        """
        
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the basis_next_expect function to each row of the data matrix
        BNE_vec = np.array([self.basis_next_expect(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])
       
        
        return BNE_vec
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.

        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare.

        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements are True if action 'a' matches
        ind = vec == a
        return ind

    def basis_next_expect(self, x, a, h_x):
        """
        Calculate the conditional expectation of the basis functions for the next state
        given the current state-action pair, using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : array-like, shape (p,)
            The current state (conditioning variable).
        a : scalar
            The action label to condition the expectation on.
        h_x : float or array-like, shape (p,)
            The bandwidth parameter(s) for the Gaussian kernel.
        basis : object
            The basis object used to compute basis functions, providing the basis functions 
            and their computation method.

        Returns
        -------
        BNE : ndarray
            The estimated conditional expectation value(s) for the given state-action pair.
        """
        
        # Compute the normalized pairwise distances between the input state-action pair and the stored training data
        u_x = (x - self.X_) / h_x # Scale differences for states by bandwidth
        
        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # Binary indicator array for action `a`
        
        # Multiply the Gaussian kernel weights by the action indicator for joint state-action smoothing
        K= K_x * ind_A
        #K+=self.alpha # Adding self.alpha ensures regularization to prevent overfitting

        # Compute the numerator: weighted sum of the basis function values for the next state
        BNE_num = np.sum(self.basis_next_df.values * K[:, np.newaxis], axis=0) # Element-wise multiplication of kernel weights
        
        # Calculate the denominator as the sum of kernel weights
        # A small constant (1e-10) is included to avoid division by zero
        BNE_denom = np.sum(K) + 1e-10
        
        # Calculate the conditional expectation by dividing the weighted sum of basis functions by the kernel weights
        BNE = BNE_num/BNE_denom
        
                      
            
        return BNE

