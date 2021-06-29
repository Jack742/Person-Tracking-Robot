clear;
clc;

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

%% Take picture and load it. Comment out to use presaved file
cam = get_camera();
img = snapshot(cam);

%connect to serial, check if works
s = connect_serial();
Angle_Move(s ,0 ,1);

%save image
file_name = "cam_img";
imwrite(img, "imgs/" + file_name + ".jpg");
size(img);


%resize image to required dimensions
target_size = [480 480 3];

%Resize image to desired dimensions
img = resized_input_img(img, target_size);

%% %Load pretrained network and Generate Config
[net, mask_subnet] = load_network(dataFolder, model_name);

%% Get network Prediction

%Make prediction
[boxes, scores, labels, masks] = predict(net, mask_subnet, img);

%Visualise the Predictions
overlayedImage = render_mask(img, boxes,labels,masks);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%dummy face tracking here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%I need to get this angle in degress from the servo start
starting_face_angle = 0;

%Set servo on closest person
Angle_Move(s ,starting_face_angle ,8);
current_angle = starting_face_angle

%%just dummy y for now
y = 100

while 1
    %%%%%%%%% GET NEW FRAME AND UPDATE Y of middle of the box %%%%%%
    %%%%% TO BE ADDED  %%%%%
    
    %capture the image
    % run your hocus pocus
    % y = 200
    
    %mAKE ACTUAL TRACKING MOVEMENT
    current_angle = track_person(s, current_angle, y)
end


%% Functions
function cam = get_camera()
    %Camera preparation
    cam = webcam('Lenovo FHD Webcam');
end

function s = connect_serial()
    serialportlist
    s = serialport("COM3",115200);
    s.Terminator;
    configureTerminator(s,"LF");
end

function next_angle = track_person(s, current_angle, y)
    %it is y index of pixels in the middle of the picture
    half_y = 1920/2;
    if y > half_y + 300
        Angle_Move(s ,current_angle - 10 ,1);
        next_angle = current_angle - 10;
    elseif y < half_y - 300
        Angle_Move(s ,current_angle + 10 ,1);
        next_angle = current_angle + 10;
    else
        next_angle = current_angle
    end
    
end

function [net, mask_subnet] = load_network(folder, model)
    %Define classes. Car is needed as it is pretrained
    pretrained = load(fullfile(folder, model));
    net = pretrained.net;
    mask_subnet = helper.extractMaskNetwork(net);
end

function [boxes,scores,labels,masks] = predict(net,mask_subnet,img)
    %Input Dimensions
    desired_image_size= [500 500 3];
    classNames = {'person', 'car','background'};
    numClasses = length(classNames)-1;
    %Create Config
    params = createMaskRCNNConfig(desired_image_size, numClasses, classNames);
    [boxes,scores,labels,masks] = detectMaskRCNN(net,mask_subnet,img,params);
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