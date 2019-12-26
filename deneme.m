data = load('son1.mat', 'sigara', 'fastRCNNLayers');
sigara = data.sigara;
fastRCNNLayers = data.fastRCNNLayers;


sigara.imageFilename = fullfile(toolboxdir('vision'),'visiondata', ...
    sigara.imageFilename);

rng(0);
shuffledIdx = randperm(height(sigara));
sigara = sigara(shuffledIdx,:);

imds = imageDatastore(sigara.imageFilename);

blds = boxLabelDatastore(sigara(:,2:end));


ds = combine(imds, blds);

ds = transform(ds,@(data)preprocessData(data,[920 968 3]));

options = trainingOptions('sgdm', ...
    'MiniBatchSize', 10, ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', 10, ...
    'CheckpointPath', tempdir);

frcnn = trainFastRCNNObjectDetector(ds, fastRCNNLayers , options, ...
    'NegativeOverlapRange', [0 0.1], ...
    'PositiveOverlapRange', [0.7 1]);


img = imread('1.jpg');


[bbox, score, label] = detect(frcnn, img);


detectedImg = insertObjectAnnotation(img,'rectangle',bbox,score);
figure
imshow(detectedImg)
