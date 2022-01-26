close all

TrainDatasetPath = fullfile('dataset','train');

% AUGMENTATION
% Also improved MaxEpoch from 8 to 15
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

    reluLayer('Name','relu_1')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_1')

    convolution2dLayer(3,16,'Padding','same','Stride', [1 1], 'Name','conv_2',...
    'WeightsInitializer',@(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    reluLayer('Name','relu_2')

    maxPooling2dLayer(2,'Stride',2,'Name','maxpool_2')

    convolution2dLayer(3,32,'Padding','same','Stride', [1 1], 'Name','conv_3',...
    'WeightsInitializer', @(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    reluLayer('Name','relu_3')

    fullyConnectedLayer(15,'Name','fc_1',...
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
    'MaxEpochs', 15,...
    'ExecutionEnvironment','parallel',...
    'Plots','training-progress'...
);

aunet = trainNetwork(auimds,layers,options);


TestDatasetPath = fullfile('dataset','test');
imdsTest = imageDatastore(TestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);

YPredicted = classify(aunet,imdsTest);
YTest = imdsTest.Labels;

figure
plotconfusion(YTest,YPredicted)