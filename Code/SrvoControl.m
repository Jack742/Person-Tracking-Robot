%%Servo can be cotrolled by sending microsecounds pulse width over USB. 
%%Full rotation is around 175 us
%%Limit urself to 1100-1900 us to have good accuracy 

clear;
clc;

serialportlist
s = serialport("COM4",115200);
s.Terminator
configureTerminator(s,"LF")

write(s,"1200","uint8");
pause(3);
write(s,"1800","uint8");
pause(3);


clear;