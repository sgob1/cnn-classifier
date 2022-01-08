close all

TrainDatasetPath = fullfile('dataset','train');

% ENSAMBLE BASELINE

imds = imageDatastore(TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(x)imresize(imread(x),[64 64]);
EnsambleNum = 10;
for i = 1:EnsambleNum
    trainQuota=0.85;
    [imdsTrain,imdsValidation] = splitEachLabel(imds,trainQuota,'randomize');

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

    %lgraph = layerGraph(layers); % to run the layers need a name
    %analyzeNetwork(lgraph)
    InitialLearningRate = 0.001;
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', InitialLearningRate, ...
        'ValidationData',imdsValidation, ...
        'MiniBatchSize',32, ...
        'MaxEpochs', 8,...
        'ExecutionEnvironment','parallel'...
        );

    baseline(i) = trainNetwork(imdsTrain,layers,options);

    
end

TestDatasetPath = fullfile('dataset','test');
imdsTest = imageDatastore(TestDatasetPath, ...
        'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);
YTest = imdsTest.Labels;
for i=1:EnsambleNum
    YPred = classify(baseline(i),imdsTest);
    for j=1:size(YPred)
        YPredicted(j,i) = YPred(j);
    end
end    


categ = categorical(categories(YTest));
for i=1:size(YPred)
    [v, argmax] = max(countcats(YPredicted(i,:)));
    predicted_output(i) = categ(argmax);
end

accuracy = sum(predicted_output == YTest)/numel(YTest);
figure
plotconfusion(YTest, predicted_output')

sum = 0;
for i=1:EnsambleNum
    YPred = classify(baseline(i),imdsTest);
    accuracy_single(i) = sum(YPred == YTest)/numel(YTest);
    sum = sum + accuracy_single(i);
end

accuracy_average = sum / EnsambleNum;