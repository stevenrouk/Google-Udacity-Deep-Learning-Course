# Google Udacity Deep Learning Course

Woohoo!

## Lesson 1 Notes

...

## Lesson 2 Notes

So we have a linear model:

```W*X + b = Y```

Let's say X is one of the images that we're feeding in, to see what letter it is. (I.e., we're trying to predict the letter given the intensity of the pixels.)

W and b are arrays full of parameters that are used to transform our raw data into Y, our logits. ('Logits' are like pre-predictions. We use the logits to create probabilities for each class -- each letter.)

Since W and b are arrays, they have many parameters in them that can be tuned in order to create accurate predictions. How many parameters? Let's figure it out...

X is a 28x28 array of image pixels that we're feeding in.

There are ten possible output classes for Y in our case, the letters 'A' through 'J'. So Y is a 10x1 array.

So, we start with a 28x28 array, and we need to multiply by W and add b to get a 10x1 array. Adding arrays together doesn't change the dimensions, so it's really W that's going to be doing our transformation here.

Matrix multiplication follows this rule for calculating input and output dimensions:

```mxn * nxr = mxr```

Notice that the 'inside' dimensions need to be the same, and the new array has the shape of the 'outside' dimensions.

So we need...

```mx28 * 28x28 =? 10,1```

This isn't going to work, because we're going to get an output array of mx28. We can get around this by 'flattening' our 28x28 array into an 784x1 array, which conveniently is done via X.flatten(). (784 is 28 squared -- we're essentially just laying all of the rows end to end.)

So now our flattened input image array is 784x1. Let's try again:

```mx784 * 784*1 = 10,1```

So our m needs to be 10 in order to get the right output dimensions.

```10x784 * 784*1 = 10,1```

Perfect! And since we're adding b after multiplying W and X, b will be the same dimensions as our final output, 10x1. Here are our dimensions for the arrays:

```python
W.shape == (10,784)
X.shape == (784,1)
b.shape == (10,)
Y.shape == (10,)
```

Awesome!

(Note: Matrix multiplication isn't commutative, so we need to multiply W*X in that order -- not X*W. If we wanted to switch the order, we could just flip the dimensions of X and W. For X, it's as simple as flattening the array and then taking the transpose via np.transpose or even just X.T.)

Here's some example code:

```python
import numpy as np
X = np.random.rand(28,28)    #fake image
X_flatten = X.flatten()      #now a 781x1 array
W = np.random.rand(10,784)   #array of 7840 parameters
b = np.random.rand(10)       #array of 10 parameters

Y = np.dot(W,X_flatten) + b
```

(Note: need to dig into array dimensions a little more to understand how they flip between one and two dimensions, such as becoming a '10,' shaped array vs. a '1,10' or '10,1' shaped array.)