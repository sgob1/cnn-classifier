clear all;

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

featuresTrain = activations(net,imdsTrain,'fc7','OutputAs','rows');
YTrain = imdsTrain.Labels;
classifier = fitcecoc(featuresTrain, YTrain);

TestDatasetPath = fullfile('dataset','test');
imdsTest = imageDatastore(TestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(cat(3, imread(x), imread(x), imread(x)), [227 227]);
YTest = imdsTest.Labels;
featuresTest = activations(net,imdsTest,'fc7', 'OutputAs','rows');
YPredicted = predict(classifier, featuresTest);
figure
plotconfusion(YTest,YPredicted)