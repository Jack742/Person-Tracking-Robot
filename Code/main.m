%% Adapted from https://uk.mathworks.com/help/vision/ug/example-InstanceSegmentationUsingMaskRCNNDeepLearningExample.html
%% Paths
dataFolder = 'models/mrcnn';
model_name = 'maskrcnn_pretrained_person_car.mat';

%%

%%%%%%%%%%%%%%%%%%%%%%%%%%####IMPORTANT#####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                         ##################                              %
%      UNCOMMENT THE BELOW CODE FOR JUST THE FIRST TIME YOU RUN THIS.     %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%trainedMaskRCNN_url = 'https://www.mathworks.com/supportfiles/vision/data/maskrcnn_pretrained_person_car.mat';
%helper.downloadTrainedMaskRCNN(trainedMaskRCNN_url,dataFolder);



%% Load Image and Resize it

img = imread('imgs/chefs.jpg');

%resize image to required dimensions
target_size = [700 700 3];

%Resize image to desired dimensions
img = resized_input_img(img, target_size);

%% %Load pretrained network and Generate Config
net, mask_subnet = load_network(dataFolder, model_name);

%% Get network Prediction

%Make prediction
[boxes,scores,labels,masks] = detectMaskRCNN(net,maskSubnet,img,params);

%Visualise the Predictions
overlayedImage = render_mask(img, boxes, labels,masks);


%% Functions
function cam = get_camera()
    %Serial port configuration
    %This makes sure that the computer USB port speaks the same convention as
    %the port on the Arduino
    serialportlist
    s = serialport("COM3",115200);
    s.Terminator;
    configureTerminator(s,"LF");

    %Camera preparation
    cameras = webcamlist;
    cam = webcam;
end

function [net, mask_subnet] = load_network(folder, model)
    %Input Dimensions
    desired_image_size= [800 800 3];
    %Define classes. Car is needed as it is pretrained
    classNames = {'person', 'car','background'};
    numClasses = length(classNames)-1;
    %Create Config
    params = createMaskRCNNConfig(desired_image_size, numClasses, classNames);
    pretrained = load(fullfile(folder, model));
    net = pretrained.net;
    maskSubnet = helper.extractMaskNetwork(net);
end

function img = resized_input_img(img, target_size)
    if size(img,1) > size(img,2)
        img = imresize(img, [target_size(1) NaN]);
    else
        img = imresize(img, [NaN target_size(2)]);
    end
end

function overlayedImage = render_mask(img, boxes, labels, masks)
    if(isempty(masks))
        overlayedImage = img;
    else
        overlayedImage = insertObjectMask(img,masks);
    end
    imshow(overlayedImage);
    showShape("rectangle",gather(boxes),"Label",labels,"LineColor",'r')
end