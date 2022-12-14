TrainDatasetPath = fullfile('dataset','train');

imds = imageDatastore(TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(x)imresize(cat(3, imread(x), imread(x), imread(x)), [227 227]);
trainQuota=0.85;
[imdsTrain,imdsValidation] = splitEachLabel(imds,trainQuota,'randomize');
aug = imageDataAugmenter("RandXReflection",true);
imageSize = [227 227 3];
auimds = augmentedImageDatastore(imageSize,imdsTrain,'DataAugmentation',aug);

net = alexnet;

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

layers(1:end-3) = freezeWeights(layers(1:end-3));

lgraph = layerGraph(layers); 
analyzeNetwork(lgraph)

options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',3, ...
    'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(imdsTrain,layers,options);


TestDatasetPath = fullfile('dataset','test');
imdsTest = imageDatastore(TestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(cat(3, imread(x), imread(x), imread(x)), [227 227]);

YPredicted = classify(netTransfer,imdsTest);
YTest = imdsTest.Labels;

figure
plotconfusion(YTest,YPredicted)


function layers = freezeWeights(layers)
% layers = freezeWeights(layers) sets the learning rates of all the
% parameters of the layers in the layer array |layers| to zero.

for ii = 1:size(layers,1)
    props = properties(layers(ii));
    for p = 1:numel(props)
        propName = props{p};
        if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
            layers(ii).(propName) = 0;
        end
    end
end

end
