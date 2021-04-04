# Typescript port of the tfjs samples



Iris 

- loss and val_loss are what nn / model is trying to minimized to ensure better prediction. loss is different from val_loss as loss is based on train set while val_loss is based on test set. 

Having low loss with a high val_loss is an indication of over-fitting whereby the model did so well against training set but poorly when it comes to un-seen dataset. 

You can see from the example here : 
https://storage.googleapis.com/tfjs-examples/iris/dist/index.html

About the model 

We have a 2 layer model that will generate 3 output (aa defined in layer 2)

const model = tf.sequential();

### if u look at inputShape, data shape is 4 column namely petal length, petal width, sepal length and sepal width. 

### define a weight 0-10 ratio in our result. For example, we have 3 different result in the end, and probabiltiy that it is any of the 3 orchid can be weight as follows. 

probabiity it is sentosa - 3, 3, 4 (which make up the 10 unit)

# we using sigmoid
model.add(tf.layers.dense(
      {units: 10, activation: 'sigmoid', inputShape: [xTrain.shape[1]]}));



### next we defined our layer to be 'softmax' and will output 3 results 
### sentosa, virginia or versicolor

model.add(tf.layers.dense({units: 3, activation: 'softmax'}));


model.summary();
