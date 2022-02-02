clear all

TrainDatasetPath = fullfile('dataset','train');

imds = imageDatastore(TrainDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imds.ReadFcn = @(x)imresize(imread(x),[64 64]);

EnsambleNum = 5;
for i = 1:EnsambleNum
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

        maxPooling2dLayer(2,'Stride',2,'Name','maxpool_3')

        fullyConnectedLayer(256,'Name','fc_1',...
        'WeightsInitializer', @(sz) randn(sz)*0.01,...
        'BiasInitializer', @(sz) zeros(sz))
        reluLayer('Name','relu_4')

        dropoutLayer(.25, 'Name', 'dropout_1')

        fullyConnectedLayer(15,'Name','fc_2',...
        'WeightsInitializer', @(sz) randn(sz)*0.01,...
        'BiasInitializer', @(sz) zeros(sz))

        softmaxLayer('Name','softmax')
        classificationLayer('Name','output')
        ];

    InitialLearningRate = 0.001;
    options = trainingOptions('sgdm', ...
        'InitialLearnRate', InitialLearningRate, ...
        'ValidationData',imdsValidation, ...
        'MiniBatchSize',32, ...
        'MaxEpochs', 25,...
        'ExecutionEnvironment','parallel'...
        );

    disp("------------------------------")
    disp("Beginning learning of network:")
    disp(i)
    net(i) = trainNetwork(auimds,layers,options);
end

TestDatasetPath = fullfile('dataset','test');
imdsTest = imageDatastore(TestDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');
imdsTest.ReadFcn = @(x)imresize(imread(x),[64 64]);
YTest = imdsTest.Labels;

% Evaluating ensamble network accuracy
for i=1:EnsambleNum
    YPred = classify(net(i),imdsTest);
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

% Evaluating average of single networks accuracy
sums = 0;
for i=1:EnsambleNum
    YPred = classify(net(i),imdsTest);
    accuracy_single(i) = sum(YPred == YTest)/numel(YTest);
    sums = sums + accuracy_single(i);
end

accuracy_average = sums / EnsambleNum;
disp(accuracy_average);
