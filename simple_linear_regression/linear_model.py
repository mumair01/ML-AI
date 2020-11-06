"""
Implementation of a simple linear regression classifier from scratch using 
tensorflow.
By: Muhammad Umair, Date: 11/5/2020
"""
# Standard library packages 
from typing import Tuple 
# Third party libraries 
import tensorflow as tf
# Local packages 
from loss import Loss


class LinearRegression:

    def __init__(self, initializer : str = "random"):
        """
        Args:
            initializer (str): Defines the type of weight and bias 
                initializations. Can be one of 'random', 'ones' or 'zeros'
        """
    
        initializers = ("random",'ones','zeros')
        self.optimizers = ("SGD")
        self.loss_types = ("MSE","SSE")
        assert initializer in initializers
        self.loss = Loss()
        # Model parameters 
        self.w = None 
        # Initializing bias
        if initializer == "ranodm":
            self.b = tf.Variable(tf.random.uniform(shape = ()),dtype = tf.float32)
        elif initializer == "ones":
            self.b = tf.Variable(1,dtype = tf.float32)
        else:
            self.b = tf.Variable(0,dtype = tf.float32)
        # State vars.
        self.trained = False  

    ################################# PUBLIC METHODS ########################

    def train(self, x : tf.Tensor, y : tf.Tensor , lamb : tf.Tensor, 
              learning_rate : int, epochs : int, optimizer : str,
              loss_type : str):
        """
        Trains a linear regression model
        
        Args:
            x (tf.Tensor): Inputs of shape (n,d)
            y (tf.Tensor): Outputs of shape (n,)
            lamb (tf.Tensor): Regularization term of shape ()
            learning_rate (float): learning rate for the optimizer b/w 0 and 1
            epochs (int): Number of training epochs
            optimizer (str): Optimizer to be used. Only SGD supported currently.
            loss_type (str): Type of the loss function. Either MSE or SSE

        Returns:
            w (tf.Tensor): Weight matrix of shape (d,)
            b (tf.Tensor): Bias value of shape ()
        """
        assert 0 <= learning_rate <= 1
        assert optimizer in self.optimizers      
        assert loss_type in self.loss_types
        self.w = tf.Variable(tf.random.uniform(
                shape = [tf.shape(x)[1]]), dtype = tf.float32)
        optimizer = self._get_optimizer(optimizer,learning_rate)
        self.w,self.b =  self._update_weights(
            x,y,self.w,self.b,lamb,learning_rate,epochs,optimizer,loss_type)
        self.trained = True 
        return self.w, self.b
        
    def fit(self, x : tf.Tensor) -> tf.Tensor:
        """
        Fits the given input using the trained model
        
        Args:
            x (tf.Tensor): Inputs of shape (n,d)
        
        Returns:
            (tf.Tensor): Outputs of shape (n,)
        """
        if self.trained:
            return self._regression_function(x,self.w,self.b) 
    
    ################################# PRIVATE METHODS ########################
    
    def _loss_SSE(self, y : tf.Tensor, y_hat : tf.Tensor):
        """
        Calculates the mean square error which is defined as:
        sum(y-y_hat)^2 / N
        
        Args:
            y (tf.Tensor): actual y value of shape (n,)
            y_hat (tf.Tensor): predicted y vlalue of shape (n,)
            
        Returns:
            (tf.Tensor): Loss value of shape ()
        """
 
        return tf.reduce_sum(tf.square(y - y_hat))
    
    def _regression_function(self, x : tf.Tensor , w : tf.Tensor, b : tf.Tensor) \
            ->tf.Tensor:
        """
        Calculates the result of the linear function y = x * w + b
        
        Args:
            x (tf.Tensor): Inputs of shape (n,d)
            w: (tf.Tensor): Weights of shape (d,)
            b: (tf.Tensor): Bias of shape ()
            
        Returns:
            (tf.Tensor): Result of linear function of shape 
        """
        return tf.reduce_sum(w * x,1) + b
    
    def _update_weights(self,x : tf.Tensor, y :  tf.Tensor, w :  tf.Tensor,
                        b :  tf.Tensor, lamb : tf.Tensor,  learning_rate : float, 
                        epochs : int, optimizer : tf.keras.optimizers,
                        loss_type : str) \
                            -> Tuple[tf.Tensor,tf.Tensor]:
        """
        Updates weights and bias based on the loss function and optimizer used
        
        Args:
            x (tf.Tensor): Inputs of shape (n,d)
            y (tf.Tensor): actual y value of shape (n,)
            w: (tf.Tensor): Weights of shape (d,)
            b: (tf.Tensor): Bias of shape ()
            lamb (tf.Tensor): Regularization term of shape ()
            learning_rate (float): learning rate for the optimizer b/w 0 and 1
            epochs (int): Number of training epochs
            optimizer (tf.keras.optimizers): Instance of a valid optimizer 
        
        Returns:
            w (tf.Tensor): Updated weight matrix of shape (d,)
            b (tf.Tensor): Updated bias of shape ()
        """
        assert 0 <= learning_rate <= 1
        for _ in range(epochs):
            with tf.GradientTape() as gt:
                gt.watch([w,b])
                y_hat = self._regression_function(x,w,b)
                # Determining loss 
                regularization_term = (
                    lamb * tf.matmul(tf.transpose(
                        tf.expand_dims(w, 1)), tf.expand_dims(w, 1)))
                loss = self._get_loss(loss_type,y,y_hat) + regularization_term 
            # Determine the gradients and the value based on the loss 
            dl_dw, dl_db = gt.gradient(loss, [w, b])
            # Update w,b based using optimizer.
            optimizer.apply_gradients(zip([dl_dw,dl_db], [w, b]))
        return (w,b)
    
    def _get_optimizer(self, optimizer : str, learning_rate : float) \
            -> tf.keras.optimizers:
        """
        Returns an instance of the appropriate optimizer.
        
        Args:
            optimizer (str): Name of allowed optimizer 
            learning_rate (float): learning rate for the optimizer b/w 0 and 1
            
        Returns:
            (tf.keras.optimizers): Instance of the appropriate optimizer.
        """
        assert optimizer in self.optimizers
        assert 0 <= learning_rate <= 1
        if optimizer == "SGD":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate)
        
    def _get_loss(self, loss_type : str, y : tf.Tensor, y_hat : tf.Tensor) \
            -> tf.Tensor:
        if loss_type == "MSE":
            return self.loss.mean_square_error(y,y_hat)
        elif loss_type == "SSE":
            return self.loss.sum_of_squared_errors(y,y_hat)
        else:
            raise Exception 
        