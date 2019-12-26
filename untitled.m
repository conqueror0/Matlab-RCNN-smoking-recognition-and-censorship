%%load gTruth.mat ;
%%data = load('son1.mat');

pp = alexnet ;
pp1= pp.Layers;

pp= pp.Layers(1:19);

ppp = [pp
    fullyConnectedLayer(2)
    softmaxLayer()
    classificationLayer()];

options = trainingOptions('sgdm', ...
    'InitialLearnRate',1e-3, ...
    'MaxEpochs',2, ...
    'CheckPointPath', tempdir);

train1 = trainFastRCNNObjectDetector(gTruth, ppp , options, ...
    'NegativeOverlapRange', [0 0.1], ...
    'PositiveOverlapRange', [0.5 1], ...
    'SmallestImageDimension', 300);


resim = imread('a1.jpg');
resim = imresize(resim, [227,227]) ;
[bbox, score , label] = detect(train1,resim);

detect = insertShape(resim ,'Rectangle', bbox);
figure 
imshow(detect)
