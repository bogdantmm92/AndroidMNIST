# AndroidMNIST

This repo cotains an improved model for DIGIT detection compared to the one offered by google. 
I think the reason why the standard model trained only on MNIST data doesn't work that well in practice is because the Android images have a different distribution compared to the MNIST ones.


To train the model we used transfer learning without freezing the parameters. 
Initially we trained the model on the mnist database and after that we trained it on around 1000 examples created from the Android app.
You can find the training dataset in the assets folder.

This repo also contains some code that generates training data, useful in case that you want to add more training data for your model.


[Here](https://colab.research.google.com/drive/1sXsuHNbXhGNF4V2p4gKmp2W-lZiXWqGi?usp=sharing) you can find the python/tensorflow code used to train the model.
