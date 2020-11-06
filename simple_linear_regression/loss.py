# Third party packages 
import tensorflow as tf 

class Loss:
    
    def __init__(self):
        pass   
    
    def mean_square_error(self, y : tf.Tensor, y_hat : tf.Tensor):
        """
        Calculates the mean square error which is defined as:
        sum(y-y_hat)^2 / N
        
        Args:
            y (tf.Tensor): actual y value of shape (n,)
            y_hat (tf.Tensor): predicted y vlalue of shape (n,)
            
        Returns:
            (tf.Tensor): Loss value of shape ()
        """
        return tf.reduce_mean(tf.square(y - y_hat))
    
    def sum_of_squared_errors(self, y : tf.Tensor, y_hat : tf.Tensor):
        """
        Calculates the mean square error which is defined as:
        sum(y-y_hat)^2
        
        Args:
            y (tf.Tensor): actual y value of shape (n,)
            y_hat (tf.Tensor): predicted y vlalue of shape (n,)
            
        Returns:
            (tf.Tensor): Loss value of shape ()
        """
        return tf.reduce_sum(tf.square(y - y_hat))
    
        
    
