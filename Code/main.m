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

%% Take picture and load it.
cam = get_camera();
img = snapshot(cam);

%save image
file_name = "cam_img";
imwrite(img, "imgs/" + file_name + ".jpg");
size(img);


%resize image to required dimensions
target_size = [480 480 3];

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
    %Camera preparation
    cameras = webcamlist;
    cam = webcam;
end

function [net, mask_subnet] = load_network(folder, model)
    %Input Dimensions
    desired_image_size= [500 500 3];
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