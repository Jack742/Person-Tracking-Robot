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



%% initialise camera and serial ports
cam = get_camera();

%connect to serial, check if works
 s = connect_serial();
 Angle_Move(s ,0 ,3);

%save image
%file_name = "cam_img";
%imwrite(img, "imgs/" + file_name + ".jpg");



%% Image size required dimensions
target_size = [480 480 3];

%% Load pretrained network and Generate Config
[net, mask_subnet] = load_network(dataFolder, model_name);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Look Around %%%%%%%%%%%%%%%%

first_spin_largest_box = [0,0,0,0];
img_number = -1;
largest_area = 0;
count = 0;

Angle_Move(s ,0 ,7);
img = snapshot(cam);
%figure;
%imshow(img)
file_name = "0";
% folderpath = "Pictures/";
% imwrite(img, folderpath + file_name + ".jpg");

img = resized_input_img(img, target_size);
[boxes, scores, labels, masks] = predict(net, mask_subnet, img);
[person_boxes, person_labels, person_masks, largest_box] = get_largest_box(boxes, labels,masks);

if length(person_labels) > 0
    [first_spin_largest_box, img_number, largest_area] = get_largest(first_spin_largest_box, largest_box,img_number,largest_area, 0);
end

Angle_Move(s ,90 ,7);
img = snapshot(cam);
%figure;
%imshow(img)
file_name = "90";
% folderpath = "Pictures/";
% imwrite(img, folderpath + file_name + ".jpg");
img = resized_input_img(img, target_size);
[boxes, scores, labels, masks] = predict(net, mask_subnet, img);
[person_boxes, person_labels, person_masks, largest_box] = get_largest_box(boxes, labels,masks);
overlayedImage = render_mask(img, person_boxes,person_labels,person_masks);
if length(person_labels) > 0
    [first_spin_largest_box, img_number, largest_area] = get_largest(first_spin_largest_box, largest_box,img_number,largest_area, 90);
end


Angle_Move(s ,180 ,7);
img = snapshot(cam);
%figure;
%imshow(img)
file_name = "180";
% folderpath = "Pictures/";
% imwrite(img, folderpath + file_name + ".jpg");
img = resized_input_img(img, target_size);
[boxes, scores, labels, masks] = predict(net, mask_subnet, img);
[person_boxes, person_labels, person_masks, largest_box] = get_largest_box(boxes, labels,masks);
overlayedImage = render_mask(img, person_boxes,person_labels,person_masks);
if length(person_labels) > 0
    [first_spin_largest_box, img_number, largest_area] = get_largest(first_spin_largest_box, largest_box, img_number,largest_area, 180);
end


Angle_Move(s ,270 ,7);
img = snapshot(cam);
%figure;
%imshow(img)
file_name = "270";
% folderpath = "Pictures/";
% imwrite(img, folderpath + file_name + ".jpg");

img = resized_input_img(img, target_size);
[boxes, scores, labels, masks] = predict(net, mask_subnet, img);
[person_boxes, person_labels, person_masks, largest_box] = get_largest_box(boxes, labels,masks);
overlayedImage = render_mask(img, person_boxes,person_labels,person_masks);
if length(person_labels) > 0
    [first_spin_largest_box, img_number, largest_area] = get_largest(first_spin_largest_box,largest_box, img_number,largest_area, 270);
end

disp("image number: ")
img_number
disp("y val: ")
first_spin_largest_box(1) +  (first_spin_largest_box(3)/2)



%%%%% JACK's hocus pokus %%%%%%
%need biggest boxes y and at which picture it is




%Closes person angle is :





%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%dummy face tracking here
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%I need to get this angle in degress from the servo start
starting_face_angle = 0;

%Set servo on closest person
Angle_Move(s ,starting_face_angle ,8);
current_angle = starting_face_angle

%%%
%NEED TO ADD THE FIRST ROTATION AND PERSON COUNTING HERE
%%%

%do spin
%if it sees someone, save the image

disp("Starting While Loop")
while 1
%     %%%%%%%%% GET NEW FRAME AND UPDATE Y of middle of the box %%%%%%
%     %%%%% TO BE ADDED  %%%%%
%     
%     %mAKE ACTUAL TRACKING MOVEMENT
    
    %% Take Image and resize to desired dimensions
    img = snapshot(cam);
    img = resized_input_img(img, target_size);
    %% Get network Prediction
    %Make prediction
   % disp("making prediction")
    [boxes, scores, labels, masks] = predict(net, mask_subnet, img);
    %disp("Getting Largest Box")
    %bounding box property order: [left, top, width, height]
    [person_boxes, person_labels, person_masks, largest_box] = get_largest_box(boxes, labels,masks);
    %get centre_y coord of largest person
    
    y = largest_box(1) + (largest_box(3)/2);
    if length(person_labels) > 0
        %disp("Tracking Person")
        current_angle = track_person(s, current_angle, y);
    end
    %Visualise the Predictions  
    %overlayedImage = render_mask(img, person_boxes,person_labels,person_masks);
end

%% Functions
function [largest_box, img_number, largest_area] = get_largest(first_spin_largest_box,largest_box, old_img, largest_area, curr_number)
    img_area = largest_box(3)*largest_box(4);
    if  img_area > largest_area
        disp("New Box!")
        first_spin_largest_box = largest_box;
        img_number=curr_number;
        largest_area = img_area;
    else
        img_number = old_img;
    end
end

function [person_boxes, person_labels, person_masks, largest_box] = get_largest_box(boxes, labels, masks)
    %% Count People and get closest
    person_boxes =  [];
    person_labels = [];
    person_masks = [];
    %area 
    largest_box = [0, 0, 0, 0];
    
    if length(labels) > 0
        disp("Found Person!")
        if length(labels) == 1
            largest_box = boxes;
            person_boxes = boxes;
            person_labels = labels;
            person_masks = masks;
        else
            for i = 1:length(labels)
                size(masks(:,:,i))
                if labels(i) == "person"
                    person_boxes = [person_boxes; boxes(i,:)];
                    person_labels = [person_labels, "person"];
                    person_masks = cat(3, person_masks, masks(:, :, i));
                    box_area = boxes(i,3) * boxes(i,4);
                    if box_area >= (largest_box(3)*largest_box(4))
                        largest_box = boxes(i, :);
                    end
                end
            end
        end
        person_masks = logical(person_masks);
    end
end

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
    half_y = 480/2;
    speed = 10;
    centre_bounds = 20;
    if y > half_y + centre_bounds
        Angle_Move(s ,current_angle + speed ,1);
        next_angle = current_angle + speed;
    elseif y < half_y - centre_bounds
        Angle_Move(s ,current_angle - speed ,1);
        next_angle = current_angle - speed;
    else
        next_angle = current_angle;
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
    desired_image_size= [800 800 3];
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
    figure
    imshow(overlayedImage);
    showShape("rectangle",gather(boxes),"Label",labels,"LineColor",'r')
end