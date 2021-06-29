clear;
clc;

%Serial port configuration
%This makes sure that the computer USB port speaks the same convention as
%the port on the Arduino
serialportlist
s = serialport("COM3",115200);
s.Terminator;
configureTerminator(s,"LF");

%Camera preparation
cameras = webcamlist
cam = webcam(2)

%hocus pocus
%need starting angle here and updates on face positions

%Move to the place of the first angle
starting_face_angle = 0


Angle_Move(s ,starting_face_angle ,10);
current_angle = starting_face_angle

%take picture at know person location
img = snapshot(cam);
imshow(img);


x = 400
y = 1800

img(x-30:x+30,y-30: y+30,:) = 0;
imshow(img);

half_y = 1920/2;

while 1
    if y > half_y + 300
        Angle_Move(s ,current_angle - 10 ,1);
        current_angle = current_angle - 10
    elseif y < half_y - 300
        Angle_Move(s ,current_angle + 10 ,1);
        current_angle = current_angle + 10
    end
end
    
  
    
    
    
    
    
    
    
    
