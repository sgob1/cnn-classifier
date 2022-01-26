close all

TrainDatasetPath = fullfile('dataset','train');

% IMPROVING 
% using sgdm
% added convolutional layer with size 9x9
% Added fully connected layer
% Added dropout layer
% MaxEpoch 25
% Learning rate 0.001
% Minibatch 32
% not real improvements from parameters.m

imds = imageDatastore(TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(x)imresize(imread(x),[64 64]);
trainQuota=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,trainQuota,'randomize');
aug = imageDataAugmenter("RandXReflection",true);
imageSize = [64 64 1];
auimds = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',aug);



layers = [
    imageInputLayer([64 64 1],'Name','input')
    
    convolution2dLayer(3,8,'Padding','same','Stride', [1 1], 'Name','conv_1',...
    'WeightsInitializer', @(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    batchNormalizationLayer('Name','BN_1')
    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')

    convolution2dLayer(5,16,'Padding','same','Stride', [1 1], 'Name','conv_2',...
    'WeightsInitializer',@(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    batchNormalizationLayer('Name','BN_2')
    reluLayer('Name','relu_2')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')

    convolution2dLayer(7,32,'Padding','same','Stride', [1 1], 'Name','conv_3',...
    'WeightsInitializer', @(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    batchNormalizationLayer('Name','BN_3')
    reluLayer('Name','relu_3')

    convolution2dLayer(9,32,'Padding','same','Stride', [1 1], 'Name','conv_4',...
    'WeightsInitializer', @(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    batchNormalizationLayer('Name','BN_4')
    reluLayer('Name','relu_4')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')

    fullyConnectedLayer(256,'Name','fc_1',...
    'WeightsInitializer', @(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))
    reluLayer('Name','relu_5')
    
    dropoutLayer(.25, 'Name', 'dropout_1')

    fullyConnectedLayer(15,'Name','fc_2',...
    'WeightsInitializer', @(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
];

lgraph = layerGraph(layers); % to run the layers need a name
% analyzeNetwork(lgraph)
InitialLearningRate = 0.001;
options = trainingOptions('sgdm', ...
    'InitialLearnRate', InitialLearningRate, ...
    'ValidationData',imdsValidation, ... 
    'MiniBatchSize',32, ...
    'MaxEpochs', 25,...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress'...
);

net_imp_conv = trainNetwork(auimds,layers,options);

TestDatasetPath = fullfile('dataset','test');
imdsTest = imageDatastore(TestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

YPredicted = classify(net_imp_conv,imdsTest);
YTest = imdsTest.Labels;

figure
plotconfusion(YTest,YPredicted)