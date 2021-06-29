function [last_angle] = Angle_Move(s,angle, time_wait)
%sends a signal needed to move the servo to a required angle
%it waits time_wait secounds before returning
%input supported 0 - 4*360

%Starting point
us = 1500;

us = us + floor(angle*0.463);
us_str = num2str(us);
write(s,us_str,"uint8");


log = "moved to " + us_str + "us."
pause(time_wait)
last_angle = angle;

end

