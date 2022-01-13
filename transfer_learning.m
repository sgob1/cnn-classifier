TrainDatasetPath = fullfile('dataset','train');

imds = imageDatastore(TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(x)imresize(cat(3, imread(x), imread(x), imread(x)), [227 227]);
trainQuota=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,trainQuota,'randomize');
aug = imageDataAugmenter("RandXReflection",true);
imageSize = [227 227 3];
auimds = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',aug,'ColorPreprocessing','gray2rgb');

net = alexnet;
%analyzeNetwork(net);

inputSize = net.Layers(1).InputSize;

layersAlex = net.Layers(1:end-3);

layers = [
    layersAlex
    fullyConnectedLayer(15,'Name','fc_2',...
    'WeightsInitializer', @(sz) randn(sz)*0.01,...
    'BiasInitializer', @(sz) zeros(sz))

    softmaxLayer('Name','softmax')
    classificationLayer('Name','output')
    ];

lgraph = layerGraph(layers); % to run the layers need a name
%analyzeNetwork(lgraph)

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(imdsTrain,layers,options);