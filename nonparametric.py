#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from . import pd, np, BaseEstimator


# In[ ]:


class est_r_pi(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression for estimating a response function.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which determines the smoothness of the estimate.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting the model.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with the input data X_.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth # Initialize the bandwidth parameter for the kernel
        
    def fit(self, X, R):
        """
        Fit the model using input data X and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        R : ndarray of shape (n_samples,)
            The response values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.X_ = X # Store input data
        self.R_ = R # Store response values
        return self

    def nw_est(self, x, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.

        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel, controlling the smoothness of the estimate.

        Returns
        -------
        nw : float
            The estimated value at point x based on the weighted average of the response values.
        """
        
        # Calculate the pairwise differences between `x` (current state) and the fitted states `self.X_`
        u_x = (x - self.X_) / h_x # Normalized difference between the input and stored states
        
        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor

        
    
        # Calculate the numerator (weighted sum of R_) and denominator (sum of weights)
        nw_num = np.sum(self.R_ * K_x)
        
        nw_denom = np.sum(K_x) + 1e-10 # Adding a small value to avoid division by zero 
        
        # Estimate the function value at x by taking the ratio of the weighted sum and the sum of weights
        nw = nw_num/nw_denom
            
            
        return nw
    
    def __call__(self, data_matrix):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the nw_est method to each row in data_matrix
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, self.bandwidth)
        return nw_est_vec
    


# In[ ]:


class est_r_pi_w(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, controlling the smoothness of the estimate.

    alpha : float, optional, default=0.1
        The regularization parameter to enhance numerical stability and prevent underflow in density calculations.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting the model.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with the input data X_.

    w : int
        The window parameter that determines the lag between the response and the input data. It should be set during model fitting.

    R_w : ndarray of shape (n_samples - w,)
        The truncated response values aligned with the window parameter, used for estimating the current state based on previous observations.

    X_w : ndarray of shape (n_samples - w, n_features)
        The truncated input data aligned with the window parameter, excluding the last w samples to maintain consistency with R_w.
    """
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth # Initialize the bandwidth parameter for the kernel

    def fit(self, X, R, w):
        """
        Fit the model using input data X, responses R, and a window parameter w.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        R : ndarray of shape (n_samples,)
            The response values.
        w : int
            The window parameter determining the lag between X and R.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.X_ = X # Store input data
        self.R_ = R # Store response values
        self.w =w # Store the window parameter
        
        # Adjust input and response data according to the window parameter
        self.R_w = self.R_[w:] # Truncated response values
        self.X_w = self.X_[:-w] # Truncated input data
        return self
    
    def nw_est(self, x, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw : float
            The estimated value at point x.
        """
        # Compute the normalized pairwise distances between input x and the original training data X_
        u_x = (x - self.X_) / h_x # Scale the differences by bandwidth
        # Compute the normalized pairwise distances between input x and the truncated data X_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences with truncated data
        
        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor
        
        # Apply the Gaussian kernel to the pairwise differences for truncated data
        normalization_factor_w = np.prod(h_x) * (2 * np.pi)**(self.X_w.shape[1] / 2)
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / normalization_factor_w

        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * K_x_w) # Element-wise multiplication of response values with kernel weights
        
        nw_denom = np.sum(K_x) + 1e-10 # Add a small value to avoid division by zero
        
        # Calculate the Nadaraya-Watson estimate
        nw = (nw_num/nw_denom)* (len(K_x)/len(K_x_w))
        
        
        return nw
    
    def __call__(self, data_matrix):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the nw_est method to each row in data_matrix
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, self.bandwidth)
        
        return nw_est_vec
    
    


# In[ ]:


class est_r_sa(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness of the density estimation.

    alpha : float, default=0.1
        The regularization parameter to prevent overfitting and improve numerical stability.
        It adds a small positive value to the kernel weights, ensuring that the computed densities remain finite
        and stable during evaluation, especially in regions where the weight of data points may be very low.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting, representing the features of each sample.
        
    A_ : ndarray of shape (n_samples,)
        The action labels corresponding to each sample in X_, indicating the actions taken for each input sample.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_, representing the outcomes we want to estimate or predict.
    """
    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates

    def fit(self, X, A, R):
        """
        Fit the model using input data X, action labels A, and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        A : ndarray of shape (n_samples,)
            The action labels corresponding to the data.
        R : ndarray of shape (n_samples,)
            The response values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the input data (X), action labels (A), and response values (R) as instance attributes
        self.X_ = X  # Input data
        self.A_ = A  # Action labels
        self.R_ = R  # Response values
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match the specified action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare (the specific action to filter).
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector where True corresponds to matching the action.
        """
        # Create a boolean array where True indicates that the action in vec matches action 'a'
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action label to use for conditioning the estimate.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the Gaussian kernel.

        Returns
        -------
        nw : float
            The estimated value at the point x, conditioned on action 'a'.
        """
        
        # Compute the normalized pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        
        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # This vector is 1 for samples where action matches 'a', 0 otherwise
        
        # Combine the state kernel weights with the indicator for the specified action
        K = K_x*ind_A # Element-wise multiplication to apply action condition
        #K += self.alpha # Adding self.alpha ensures stability and prevents overfitting
        
        # Calculate the weighted sum of response values (numerator)
        nw_num = np.sum(self.R_ * K)  # Response values weighted by kernel weights

        # Calculate the sum of kernel weights (denominator) 
        # A small constant (1e-10) is added to avoid numerical instability
        nw_denom = np.sum(K) + 1e-10
        
        # Calculate the final estimate by dividing the numerator by the denominator
        nw = nw_num/nw_denom
        
            
        return nw
    
    def __call__(self, data_matrix, a):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a : scalar
            The action label to use for conditioning the estimate.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Apply the nw_est method to each row in data_matrix, estimating the function value for action 'a'
        nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, a, self.bandwidth)
        #nw_est_vec = np.array([self.nw_est(data_matrix[i], a, self.bandwidth) for i in range(len(data_matrix))])
        
        return nw_est_vec


# In[ ]:


class est_r_saData(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression.

    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness of the density estimation.

    alpha : float, default=0.1
        The regularization parameter to prevent overfitting and improve numerical stability.
        It adds a small positive value to the kernel weights, ensuring that the computed densities remain finite
        and stable during evaluation, especially in regions where the weight of data points may be very low.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting, representing the features of each sample.
        
    A_ : ndarray of shape (n_samples,)
        The action labels corresponding to each sample in X_, indicating the actions taken for each input sample.
        
    R_ : ndarray of shape (n_samples,)
        The response values associated with X_, representing the outcomes we want to estimate or predict.
    """
    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates

    def fit(self, X, A, R):
        """
        Fit the model using input data X, action labels A, and responses R.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data.
        A : ndarray of shape (n_samples,)
            The action labels corresponding to the data.
        R : ndarray of shape (n_samples,)
            The response values.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store the input data (X), action labels (A), and response values (R) as instance attributes
        self.X_ = X  # Input data
        self.A_ = A  # Action labels
        self.R_ = R  # Response values
        return self
    
    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match the specified action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against (action labels).
        a : scalar
            The action to compare (the specific action to filter).
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector where True corresponds to matching the action.
        """
        # Create a boolean array where True indicates that the action in vec matches action 'a'
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action label to use for conditioning the estimate.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the Gaussian kernel.

        Returns
        -------
        nw : float
            The estimated value at the point x, conditioned on action 'a'.
        """
        # Compute the normalized pairwise differences for the state and action
        u_x = (x - self.X_) / h_x # Difference between input state and fitted states
        
        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor
        
        # Create an indicator vector for samples corresponding to action 'a'
        ind_A = self.indicator(self.A_, a) # This vector is 1 for samples where action matches 'a', 0 otherwise
        
        # Combine the state kernel weights with the indicator for the specified action
        K = K_x*ind_A # Element-wise multiplication to apply action condition
        #K += self.alpha # Adding self.alpha ensures stability and prevents overfitting
        
        # Calculate the weighted sum of response values (numerator)
        nw_num = np.sum(self.R_ * K)  # Response values weighted by kernel weights

        # Calculate the sum of kernel weights (denominator) 
        # A small constant (1e-10) is added to avoid numerical instability
        nw_denom = np.sum(K) + 1e-10
        
        # Calculate the final estimate by dividing the numerator by the denominator
        nw = nw_num/nw_denom
        
  
            
        return nw
    
    def __call__(self, data_matrix, a_vec):
        """
        Apply Nadaraya-Watson kernel regression across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.
            
        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix.
        """
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the nw_est method row by row, passing the corresponding element of a_vec
        nw_est_vec = np.array([self.nw_est(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])
        return nw_est_vec


# In[ ]:


class est_r_pi_sa_w(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.
    
    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness
        of the kernel density estimation by adjusting how much influence nearby points have.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting (e.g., state variables or features). This data
        represents the independent variables in the model.
    
    A_ : ndarray of shape (n_samples,)
        The action values associated with each sample in X_. These could be either
        discrete (e.g., action labels) or continuous, depending on the problem setup.
    
    R_ : ndarray of shape (n_samples,)
        The response values (e.g., rewards or outcomes) associated with each sample
        in X_ and A_. These are the dependent variables the model aims to estimate.
    
    w : int
        The window parameter that defines the lag between the response variable and
        the input data. This lag is useful for time-dependent models where the response
        is expected to be influenced by a previous state.
    
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter. This array
        contains only the response values that have corresponding lagged input data.
    
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter. This ensures that
        each row in X_w corresponds to a response value in R_w.
    
    A_w : ndarray of shape (n_samples-w,)
        The truncated action data aligned with the window parameter. Each action in
        A_w aligns with the corresponding state in X_w and response in R_w.
    """

    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates

    def fit(self, X, A, R, w):
        """
        Fit the model using input data X, action data A, responses R, and a window parameter w.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data (e.g., states or features).
        A : ndarray of shape (n_samples,)
            The action data associated with each input (e.g., discrete or continuous actions).
        R : ndarray of shape (n_samples,)
            The response values (e.g., rewards or outcomes).
        w : int
            The window parameter determining the lag between X and R.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store input, action, and response data
        self.X_ = X  # Store the input data (states/features)
        self.A_ = A  # Store the action data corresponding to each state
        self.R_ = R  # Store the response values (rewards)
        self.w = w   # Store window parameter (lag between X and R)

        # Adjust input, action, and response data based on the window parameter
        self.R_w = self.R_[w:]      # Truncate response values by skipping the first 'w' values
        self.X_w = self.X_[:-w]     # Truncate input data to align with the window-adjusted responses
        self.A_w = self.A_[:-w]     # Truncate action data to align with the window-adjusted inputs
        return self  # Return the fitted estimator

    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements equal to action 'a' are marked as True
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action for which to estimate the value.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw : float
            The estimated value at point x for action 'a'.
        """
        
        # Compute the normalized pairwise distances between input x and the original training data X_ and A_
        u_x = (x - self.X_) / h_x  # Scale the differences by bandwidth for state
        # Compute the normalized pairwise distances between input x and the truncated data X_w and A_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences by bandwidth for truncated state

        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor
        
        # Create an indicator for the current action 'a' in the full action data (A_)
        ind_A = self.indicator(self.A_, a)
        
        # Apply the Gaussian kernel to the pairwise differences for truncated data
        normalization_factor_w = np.prod(h_x) * (2 * np.pi)**(self.X_w.shape[1] / 2)
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / normalization_factor_w
        
        # Create an indicator for the current action 'a' in the truncated action data (A_w)
        ind_A_w = self.indicator(self.A_w, a)
        
        # Combine the kernel weights with action indicator for full data and add alpha for numerical stability
        K = K_x*ind_A
        #K+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Combine the kernel weights with action indicator for truncated data and add alpha for stability
        K_w = K_x_w*ind_A_w
        #K_w+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * K_w)   # Element-wise multiplication of response values with kernel weights
        
        # Compute the denominator as the sum of kernel weights, with a small constant to avoid division by zero
        nw_denom = np.sum(K) + 1e-10
        
        # Compute the final estimate as the ratio of the weighted response sum to the sum of kernel weights
        nw = (nw_num/nw_denom) * (len(K)/len(K_w))
        
        return nw

    def __call__(self, data_matrix, a):
        """
        Apply Nadaraya-Watson kernel regression to each row in the data matrix.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a : scalar
            The action for which to estimate the values.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix, for action 'a'.
        """
        # Apply the Nadaraya-Watson estimation method to each row of the data matrix
        #nw_est_vec = np.apply_along_axis(self.nw_est, 1, data_matrix, a, self.bandwidth)
        nw_est_vec = np.array([self.nw_est(data_matrix[i], a, self.bandwidth) for i in range(len(data_matrix))])
        
        return nw_est_vec  # Return the estimated values for each data point and action


# In[ ]:


class est_r_pi_sa_wData(BaseEstimator):
    """
    Custom estimator using Nadaraya-Watson kernel regression with a window parameter.
    
    Parameters
    ----------
    bandwidth : float, default=1.0
        The bandwidth parameter for the Gaussian kernel, which controls the smoothness
        of the kernel density estimation by adjusting how much influence nearby points have.

    Attributes
    ----------
    X_ : ndarray of shape (n_samples, n_features)
        The input data used for fitting (e.g., state variables or features). This data
        represents the independent variables in the model.
    
    A_ : ndarray of shape (n_samples,)
        The action values associated with each sample in X_. These could be either
        discrete (e.g., action labels) or continuous, depending on the problem setup.
    
    R_ : ndarray of shape (n_samples,)
        The response values (e.g., rewards or outcomes) associated with each sample
        in X_ and A_. These are the dependent variables the model aims to estimate.
    
    w : int
        The window parameter that defines the lag between the response variable and
        the input data. This lag is useful for time-dependent models where the response
        is expected to be influenced by a previous state.
    
    R_w : ndarray of shape (n_samples-w,)
        The truncated response values aligned with the window parameter. This array
        contains only the response values that have corresponding lagged input data.
    
    X_w : ndarray of shape (n_samples-w, n_features)
        The truncated input data aligned with the window parameter. This ensures that
        each row in X_w corresponds to a response value in R_w.
    
    A_w : ndarray of shape (n_samples-w,)
        The truncated action data aligned with the window parameter. Each action in
        A_w aligns with the corresponding state in X_w and response in R_w.
    """

    def __init__(self, bandwidth=1.0):
        # Initialize the estimator with the specified bandwidth and regularization parameter
        self.bandwidth = bandwidth # Set the bandwidth for the kernel, influencing the smoothness of estimates

    def fit(self, X, A, R, w):
        """
        Fit the model using input data X, action data A, responses R, and a window parameter w.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            The input data (e.g., states or features).
        A : ndarray of shape (n_samples,)
            The action data associated with each input (e.g., discrete or continuous actions).
        R : ndarray of shape (n_samples,)
            The response values (e.g., rewards or outcomes).
        w : int
            The window parameter determining the lag between X and R.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Store input, action, and response data
        self.X_ = X  # Store the input data (states/features)
        self.A_ = A  # Store the action data corresponding to each state
        self.R_ = R  # Store the response values (rewards)
        self.w = w   # Store window parameter (lag between X and R)

        # Adjust input, action, and response data based on the window parameter
        self.R_w = self.R_[w:]      # Truncate response values by skipping the first 'w' values
        self.X_w = self.X_[:-w]     # Truncate input data to align with the window-adjusted responses
        self.A_w = self.A_[:-w]     # Truncate action data to align with the window-adjusted inputs
        return self  # Return the fitted estimator

    def indicator(self, vec, a):
        """
        Create an indicator vector where elements are True if they match action 'a'.
        
        Parameters
        ----------
        vec : array-like, shape (n_samples,)
            The vector to compare against.
        a : scalar
            The action to compare.
        
        Returns
        -------
        ind : array-like, shape (n_samples,)
            Indicator vector that is True where 'vec' equals 'a'.
        """
        # Create a boolean vector where elements equal to action 'a' are marked as True
        ind = vec == a
        return ind

    def nw_est(self, x, a, h_x):
        """
        Estimate the function value at a given point using Nadaraya-Watson kernel regression.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            The point at which to estimate the function value.
        a : scalar
            The action for which to estimate the value.
        h_x : float or ndarray of shape (n_features,)
            The bandwidth parameter(s) for the kernel.

        Returns
        -------
        nw : float
            The estimated value at point x for action 'a'.
        """
        
        
        # Compute the normalized pairwise distances between input x and the original training data X_ and A_
        u_x = (x - self.X_) / h_x  # Scale the differences by bandwidth for state
        # Compute the normalized pairwise distances between input x and the truncated data X_w and A_w
        u_x_w = (x - self.X_w) / h_x # Scale the differences by bandwidth for truncated state

        # Compute the Gaussian kernel weights for the current state
        # Adjusted normalization for vector-valued h_x
        normalization_factor = np.prod(h_x) * (2 * np.pi)**(self.X_.shape[1] / 2)
        K_x = np.exp(-0.5 * np.sum(u_x**2, axis=1)) / normalization_factor
        
        # Create an indicator for the current action 'a' in the full action data (A_)
        ind_A = self.indicator(self.A_, a)
        
        # Apply the Gaussian kernel to the pairwise differences for truncated data
        normalization_factor_w = np.prod(h_x) * (2 * np.pi)**(self.X_w.shape[1] / 2)
        K_x_w = np.exp(-0.5 * np.sum(u_x_w**2, axis=1)) / normalization_factor_w
        
        # Create an indicator for the current action 'a' in the truncated action data (A_w)
        ind_A_w = self.indicator(self.A_w, a)
        
        # Combine the kernel weights with action indicator for full data and add alpha for numerical stability
        K = K_x*ind_A
        #K+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Combine the kernel weights with action indicator for truncated data and add alpha for stability
        K_w = K_x_w*ind_A_w
        #K_w+= self.alpha # Regularization term for stability in case of small or zero weights
        
        # Compute the numerator: the weighted sum of truncated response values
        nw_num = np.sum(self.R_w * K_w)   # Element-wise multiplication of response values with kernel weights
        
        # Compute the denominator as the sum of kernel weights, with a small constant to avoid division by zero
        nw_denom = np.sum(K) + 1e-10
        
        # Compute the final estimate as the ratio of the weighted response sum to the sum of kernel weights
        nw = (nw_num/nw_denom) * (len(K)/len(K_w))
        
        
        
        return nw

    def __call__(self, data_matrix, a_vec):
        """
        Apply Nadaraya-Watson kernel regression across all rows in the data matrix with action vector.

        Parameters
        ----------
        data_matrix : ndarray of shape (n_samples, n_features)
            The data points at which to estimate the function values.
        a_vec : array-like, shape (n_samples,)
            The action vector for which the policy probability is estimated.

        Returns
        -------
        nw_est_vec : ndarray of shape (n_samples,)
            The estimated values for each row in the data matrix, for action 'a'.
        """
        # Ensure that data_matrix and a_vec have the same length
        assert data_matrix.shape[0] == a_vec.shape[0], "data_matrix and a_vec must have the same length"
        
        # Apply the Nadaraya-Watson estimation method to each row of the data matrix
        nw_est_vec = np.array([self.nw_est(data_matrix[i], a_vec[i], self.bandwidth) for i in range(len(data_matrix))])
        
        return nw_est_vec  # Return the estimated values for each data point and action

