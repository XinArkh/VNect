function [joint_idx, joint_parents_o1, joint_parents_o2, joint_names] = mpii_get_joint_set(joint_set_name)
                   
    all_joint_names = {'spine3', 'spine4', 'spine2', 'spine', 'pelvis', ...     %5       
        'neck', 'head', 'head_top', 'left_clavicle', 'left_shoulder', 'left_elbow', ... %11
       'left_wrist', 'left_hand',  'right_clavicle', 'right_shoulder', 'right_elbow', 'right_wrist', ... %17
       'right_hand', 'left_hip', 'left_knee', 'left_ankle', 'left_foot', 'left_toe', ...        %23   
       'right_hip' , 'right_knee', 'right_ankle', 'right_foot', 'right_toe'}; 
      
  %The O1 and O2 indices are relaive to the joint_idx, regardless of the joint set 
                   
switch joint_set_name
    case 'all'
        joint_idx = 1:28;              
        joint_parents_o1 = [3, 1, 4, 5, 5, 2, 6, 7, 6, 9, 10, 11, 12, 6, 14, 15, 16, 17, 5, 19, 20, 21, 22, 5, 24, 25, 26, 27 ];
        joint_parents_o2 = [4, 3, 5, 5, 5, 1, 2, 6, 2, 6, 9, 10, 11, 2, 6, 14, 15, 16, 4, 5, 19, 20, 21, 4, 5, 24, 25, 26];
        joint_names = all_joint_names;
        
    case 'relevant' %Human3.6m compatible joint set in Our order
        joint_idx = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7];
        joint_parents_o1 = [ 2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2];
        joint_parents_o2 = [ 16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16];
        joint_names = all_joint_names(joint_idx);
    case 'extended' %Human3.6m compatible joint set in Our order + End effectors for Hands and Feet
        joint_idx = [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7, 18, 13, 28, 23];
        joint_parents_o1 = [ 2, 16, 2, 3, 4, 2, 6, 7, 15, 9, 10, 15, 12, 13, 15, 15, 2, 5, 8, 11, 14];
        joint_parents_o2 = [ 16, 15, 16, 2, 3, 16, 2, 6, 16, 15, 9, 16, 15, 12, 15, 15, 16, 4, 7, 10, 13];
        joint_names = all_joint_names(joint_idx);
    otherwise
end
end
