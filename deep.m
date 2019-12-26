clc
clear all
close all

img=imread('1.jpg');

load imageLabelingSession

imshow(img)

sample1={'images\a.jpg';'images\b.jpg'};

sample2={[980,393,31.56];[1040,354,73,72]};

label=table(sample1,sample2)

imdir=fullfile('C:\Users\msi\Desktop\r_cnn')

options=trainingOptions('sgdm','MiniBatchSize',128,'InitialLearnRate',1e-6,'MaxEpochs',20);

train=trainRCNNObjectDetector(label,imageLabelingSession,options,'NegativeOverlapRange',[0 0.3]);


