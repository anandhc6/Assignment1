# Assignment1
This project uses Object Oriented Programming for modularity.

The core of this project is 'fit' function.

## fit(X,y,batch,epochs,eta,weight_para,optimiser,loss,weight_decay,activation_para,la)

To train the neural network,

x - training data  
y - training labels  
batch - batch size used for training  
epochs - number of epochs until which the model will be trained  
eta - learning rate  
weight_para - weight initialization method to be used. By default it is None for which weights are initialized randomly. 'Xavier' method is supported.  
optimiser - sgd, momentum, nesterov, rmsprop, adam, nadam
loss - loss function to be used . cross entropy , squared error loss
weight_decay - Activation function to be used . sigmoid , relu and tanh functions are supported.
activation_para - L2 regularization weight decay parameter. Default value is 0 which means no L2 regularization
la - contains number of neurons , hidden layers , dimension of input features and number of output classes

## yhate=feedforward(x, params,activation_para)
It will predict the classes for the given data.

X - Data for which the accuracy of the model need to be evaluated
params - contains weights and biases 
activation_para - passes the activation to be used

It returns an array of predictions

## loss_accuracy(x,params,y,t,loss,activation_para, n_class=10)
To calculate loss and accuracy of the model given some data

X - Data for which the accuracy of the model need to be evaluated
params - contains weights and biases 
y - Corresponding class labels of the data 
t - number of epochs
loss - passes the loss function to be used
activation_para - passes the activation to be used
n_class=10 - represents the output classes

Returns the accuracy score


### Example:
``` python

train_acc,train_loss,val_acc,val_loss=fit(X_train,y_train,batch,epochs,eta,weight_para,optimiser,loss,weight_decay,activation_para,la)
test_acc=accuracy(X_test.T,y_test,params)

```
The "fit" function needs to be called which will fit the required data using the model.
It returns training accuracy and loss, validation accuracy and loss.
