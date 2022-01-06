# CNN Classifier

This project requires the implementation of an image classifier based on convo lutional neural networks. The provided dataset (from [Lazebnik et al., 2006]), contains 15 categories (office, kitchen, living room, bedroom, store, industrial, tall building, inside city, street, highway, coast, open country, mountain, forest, suburb), and is already divided in training set and test set.

| # | type | size |
|---|------|------|
| 1 | Image input     |  64 x 64 x 1 images    |
| 2 | Convolution     |  8  3 x 3 convolutions with stride 1    |
| 3 | ReLU     |  |
| 4 | Max Pooling     |  2 x 2 max pooling with stride 2 |
| 5 | Convolution     |  16  3 x 3 convolutions with stride 1    |
| 6 | ReLu     |  |
| 7 | Max Pooling     |  2 x 2 max pooling with stride 2 |
| 8 | Convolution     |  32  3 x 3 convolutions with stride 1    |
| 9 | ReLu     |  |
| 10 | Fully Connected |  15 |
| 11 | Softmax |  softmax |
| 12 | Classification output |  crossentropyex    |

You are required to:

1. train a shallow network from scratch according to the following specifications:

+ use the network layout shown in Table 1:
+ since the input image is 64×64 you will need to resize the images in order to feed them to the network; follow the simple approach of rescaling the whole image independently along x and y to get the proper size (an example of such anisotropic rescaling is shown in Fig. 3)). Other approaches exist, see below about the optional improvement of data augmentation;
+ split the provided training set in 85% for actual training set and 15% to be used as validation set;
+ employ the stochastic gradient descent with momentum optimization algorithm, using the default parameters of the library you use, except for those specified in the following;
+ use minibatches of size 32 and initial weights drawn from a Gaussian distribution with a mean of 0 and a standard deviation of 0.01; set the initial bias values to 0;
+ by using a proper value for the learning rate, you should be able to obtain an overall test accuracy of around 30% 2;
+ report and discuss the plots of loss and accuracy during training, for both the training set and the validation set;
+ comment on the criterion you choose for stopping the training;
+ report the confusion matrix and the overall accuracy, both computed on the test set.

2. Improve the previous result, according to the following suggestions (not all the following will necessarily result in an increase of accuracy, but you should be able to obtain a test accuracy of about 60% by employing some of them):

+ data augmentation: given the small training set, data augmentation is likely to improve the performance. For the problem at hand, left-to-right reflections are a reasonable augmentation technique. By augmenting the train data using left-to-right reflections you should get an accuracy of about 40%;
+ batch normalization [Ioffe and Szegedy, 2015]: add batch normalization layers before the reLU layers;
+ change the size and/or the number of the convolutional filters, for instance try increasing their support as we move from input to output: 3×3, 5×5, 7×7;
+ play with the optimization parameters (learning rate, weights regularization, minibatch size . . . ); you can also switch to the adam optimizer;
+ dropout: add some dropout layer to improve regularization;
+ employ an ensemble of networks (five to ten), trained independently. Use the arithmetic average of the outputs to assign the class, as in [Szegedy et al., 2015].

Comment on any significant change you notice after the application of the previous modifications.

3. Use transfer learning based on a pre-trained network, for instance AlexNet [Krizhevsky et al., 2012], in the following two manners (in both of the cases you should get a test accuracy above 85%):

+ employ the pre-trained network as a feature extractor, accessing the activation of an intermediate layer (for instance the last convolutional layer) and train a multiclass linear SVM. For implementing the multiclass SVM use any of the approaches seen in the lectures, for instance DAG.

4. Optionally, in tasks 2 and 3, you could improve data augmentation, adding to the left-right reflection a random cropping of a sufficiently large region, followed by small rotations and rescaling (an example of crop and rescaling is shown in Fig. 3);
5. optionally, in task 2, you could add more convolutional and/or fullyconnected layers;
6. optionally, in task 3, you could employ nonlinear SVMs;
7. optionally, in task 3, you could implement the multiclass SVM using the Error Correcting Output Code approach [Dietterich and Bakiri, 1994, James and Hastie, 1998].

## Notes 

+ Carry out task 1 exactly as required, in order to obtain a baseline for the subsequent tasks;
+ apply the data augmentation only to the train fraction of the training images. In other words, do not augment the validation and test sets;
+ independent of the data augmentation techniques, for validation and test use only the anisotropic rescaling show on top of Fig. 3 (squeeze the whole image to the required size).
