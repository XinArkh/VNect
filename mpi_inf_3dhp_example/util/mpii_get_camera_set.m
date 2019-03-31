function [camera_set] = mpii_get_camera_set(camera_set_name)
 
switch camera_set_name
    
    case 'regular'  
        camera_set = 0:13; %Cameras with regular lenses, not fisheye
    case 'relevant'
        camera_set = 0:10; %All cameras except the ceiling mounted ones
	case 'ceiling'
        camera_set = 11:13;   %Top down views
    case 'vnect'
        camera_set = [0, 1, 2, 4, 5, 6, 7, 8];   %Chest high, knee high and 2 cameras angled down. Use for VNect @ SIGGRAPH 17
    case 'mm3d_chest'
        camera_set = [0, 2, 4, 7, 8]; %Subset of chest high, used in "Monocular 3D Human Pose Estimation in-the-wild Using Improved CNN supervision"
    otherwise
        camera_set = [];
end
