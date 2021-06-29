%% Adapted from https://uk.mathworks.com/help/vision/ug/example-InstanceSegmentationUsingMaskRCNNDeepLearningExample.html
%% Paths
dataFolder = 'models/mrcnn';
model_name = 'maskrcnn_pretrained_person_car.mat';
%% IMPORTANT only need to run this once

%trainedMaskRCNN_url = 'https://www.mathworks.com/supportfiles/vision/data/maskrcnn_pretrained_person_car.mat';
%helper.downloadTrainedMaskRCNN(trainedMaskRCNN_url,dataFolder);

%%
pretrained = load(fullfile(dataFolder, model_name));
net = pretrained.net;
maskSubnet = helper.extractMaskNetwork(net);

classNames = {'person', 'car','background'};
numClasses = length(classNames)-1;

img = imread('imgs/chefs.jpg');

%resize image to required dimensions
target_size = [700 700 3];

if size(img,1) > size(img,2)
    img = imresize(img, [target_size(1) NaN]);
else
    img = imresize(img, [NaN target_size(2)]);
end

imageSizeTrain = [800 800 3];
params = createMaskRCNNConfig(imageSizeTrain, numClasses, classNames);

[boxes,scores,labels,masks] = detectMaskRCNN(net,maskSubnet,img,params);

if(isempty(masks))
    overlayedImage = img;
else
    overlayedImage = insertObjectMask(img,masks);
end
imshow(overlayedImage)

showShape("rectangle",gather(boxes),"Label",labels,"LineColor",'r')


