%% Image classification using convolutional neural networks
% In this example we will employ a (shallow) convolutional neural network to 
% classify images belonging to the CIFAR10 dataset. Precisely, we will use a subset 
% (CIFAR2) containing two classes only ("automobile" and "truck"). 
%% The dataset
% The provided dataset is located in the folder |cifar2|. The folder |train| 
% contains two subfolders |automobile| and |truck|, each consisting of 5000 color 
% images of size 32x32 to be used for training. The folder |test| is organized 
% in the same way and there are 1000 test images per class.

close all force

cifar2TrainDatasetPath = fullfile('cifar2','train');

imds = imageDatastore(cifar2TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
%% 
% The object |imageDatastore| is useful to manage a collection of image files, 
% where each individual image fits in memory, but the entire collection of images 
% does not necessarily fit. An |imageDatastore| object has properties that can 
% be used, for instance, specifying tranformations that are applied to the images 
% at time of loading them into memory.

class(imds) % what kind of object is imds
%% 
% 
imds;
labelCount = countEachLabel(imds);
unique(imds.Labels)
%% 
% To read an image, use the member function |readimage|:

%{
% reading an image. 
iimage=100;
img = imds.readimage(iimage); 
figure;
imshow(img,'initialmagnification',1000)
%%
figure
imshow(preview(imds),'initialmagnification',1000); %preview(imds) is the same as imds.readimage(1)
%%
% show the first 256 images 
im=imtile(imds.Files(1:256)); 
figure
imshow(im,'initialmagnification',200)
%%
% show the last 256 images
im=imtile(imds.Files(end-255:end));
figure
imshow(im,'initialmagnification',200)
%% 

% If needed, you can define a custom function applied to each image at time 
% of reading. For instance, to resize each image to 256x256:
%}
% automatic resizing
imds.ReadFcn = @(x)imresize(imread(x),[256 256]);
%% 
% In the above code, the property |ReadFcn| is assigned a handle to an _inline 
% function_, whose argument is the filename |x|. 
% 
% To restore to default read function:

imds.ReadFcn = @(x)imread(x);
%% 
% To perform an automatic rescaling of the values use the following syntax:

% automatic rescaling
divideby=255;
imds.ReadFcn = @(x)double(imread(x))/divideby;
%% 
% Some random instances of the training set, with corresponding label:

% show some instances
%{
figure;
perm = randperm(length(imds.Labels),20);
for ii = 1:20
    subplot(4,5,ii);
    imshow(imds.Files{perm(ii)}); 
    title(imds.Labels(perm(ii)));
end
sgtitle('some instances of the training set')
%}
%% 
% In order to estimate the generalization capability during training, we need 
% to extract a valdation set from the provided training set. Let's take the 85% 
% of the images for actual training and the remaining 15% for validation.

% split in training and validation sets: 85% - 15%
quotaForEachLabel=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,quotaForEachLabel,'randomize');
%% Network design and training
% In the following we will design a basic network, train it and gradually improve 
% the design and/or the training options. For time reasons, all the trainings 
% employ just 5 epochs. Bear in mind that the proper way to train is to employ 
% early stopping based on the validation accuracy/loss.
% Basic network
% The structure of the network is as follows. 

% create the structure of the network
layers = [
    imageInputLayer([32 32 3],'Name','input') % 32x32 color images
    % To restore to default read function:

    convolution2dLayer(3,16,'Padding','same','Name','conv_1') 
    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')
    
    convolution2dLayer(3,16,'Padding','same','Name','conv_2')
    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')
    
    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')
    

    fullyConnectedLayer(256,'Name','fc_1')
    reluLayer('Name','relu_3')

    dropoutLayer(.25, 'Name', 'dropout_1')
     
    fullyConnectedLayer(10,'Name','fc_2')
    reluLayer('Name','relu_4')

    fullyConnectedLayer(2,'Name','fc_3') % 2 because we have 2 classes
    softmaxLayer('Name','softmax')
    
    classificationLayer('Name','output')];

    lgraph = layerGraph(layers); % to run the layers need a name
    analyzeNetwork(lgraph)
%% 
% %% 
% Now we set the training options. |sgdm| stands for _stochastic gradient descent 
% with momentum_. Notice that we set |ValidationPatience| to |Inf,| meaning that 
% the solver won't stop until |MaxEpochs| is reached.

    % training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',20, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ... 
    'ValidationFrequency',10, ...
    'ValidationPatience',Inf,...
    'Verbose',false, ...
    'MiniBatchSize',128, ...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress');
%% 
% Then we train the network.

% train the net
net = trainNetwork(imdsTrain,layers,options);
%% 
% % 
% The plot suggests that with more than 5 epochs we could improve the result 
% (we stopped too early). 
%% 
% However, let's evaluate the performance on the test set.

% evaluate performance on test set

cifar2TestDatasetPath  = fullfile('cifar2','test');

imdsTest = imageDatastore(cifar2TestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)double(imread(x))/divideby;

% apply the network to the test set
YPredicted = classify(net,imdsTest);
YTest = imdsTest.Labels;

% overall accuracy
accuracy = sum(YPredicted == YTest)/numel(YTest);

% confusion matrix
figure
plotconfusion(YTest,YPredicted)

% 

features_conv_1_Test = activations(net, imdsTest,'conv_1');

size(features_conv_1_Test)

image=100;
figure;
subplot(1,2,1);
imshow(imdsTest.readimage(iimage));
subplot(1,2,2);
imshow(imtile(features_conv_1_Test(:,:,:,iimage)), []);

features_relu_1_Test = activations(net, imdsTest,'relu_1');

size(features_relu_1_Test)

image=100;
figure;
subplot(1,2,1);
imshow(imdsTest.readimage(iimage));
subplot(1,2,2);
imshow(imtile(features_relu_1_Test(:,:,:,iimage)), []);


features_conv_2_Test = activations(net, imdsTest,'conv_2');

size(features_conv_2_Test)

image=100;
figure;
subplot(1,2,1);
imshow(imdsTest.readimage(iimage));
subplot(1,2,2);
imshow(imtile(features_conv_2_Test(:,:,:,iimage)), []);

w1 = net.Layers(2).Weights;
size(w1)

figure
imshow(imtile(w1(:,:,1,:),'bordersize',[1 1]), [], 'initialmagnification',3000)
pause(1)
figure
imshow(imtile(w1(:,:,2,:),'bordersize',[1 1]), [], 'initialmagnification',3000)
pause(1)
figure
imshow(imtile(w1(:,:,3,:),'bordersize',[1 1]), [], 'initialmagnification',3000)


