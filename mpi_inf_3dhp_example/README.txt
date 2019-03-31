####################
MPI-INF-3DHP Dataset
####################

Terms of use: 
The provided dataset is intended for research purposes only and any use of it for non-scientific and/or commercial means is not allowed. This includes publishing any scientific
results obtained with our data in non-scientific literature, such as tabloid press. We ask the user to respect our actors and not to use the data for any distasteful manipulations. 
If you use our training or test data, you are required to cite the origin: 
[1] VNect: Real-time 3D Human Pose Estimation With A Single RGB Camera (ACM Trans. on Graphics, SIGGRAPH 2017)
	Mehta, D.; Sridhar, S.; Sotnycheno, O.; Rhodin, H.; Shafiei, M.; Seidel, H.; Xu, W.; Casas, D.; Theobalt, C.
[2] Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision (3DV 2017)
	Mehta, D.; Rhodin, H.; Casas, D.; Fua, P.; Sotnychenko, O.; Xu, W.; Theobalt, C.

Refer to the license (license.txt) distributed with the data.

########################
Downloading the Dataset
########################
Use the script get_dataset.sh to download the training set and get_testset.sh for the test set. You would need to read and review the configuration under conf.ig before you can proceed with downloading the dataset

####################
Training Set Details
####################
The dataset comprises of 8 subjects, covering the following 8 activities with 2 sequences per subject.
Sequence 1: 
A1: Walking/Standing
	Walking, jogging, waiting in a queue, pointing at things, having an animated conversation, smoking while standing or walking, phone call etc
A2: Exercise
	Lunges, pushups, bridge, stretch legs, other forms of slow exercise
A3: Sitting(1)
	Eating, working at a computer, picking something off the floor, lie back on the chair, cross feet, sit with hands behind head etc
A4: Crouch/Reach
	Crouch and pretend lift something, tie shoe laces, photography while crouching, crouch and interact with objects etc

Sequence 2: 
A5: On the Floor
	Cycling, crunches and other complicated poses while lying on the ground.
A6: Sports
	Boxing, tennis, golf, soccer and other forms of fast motion. Slightly awkward because the green screen covering the floor didn't have traction on the floor.
A7: Sitting(2)
	Move the chair around while seated, wave someone over, cheer for sports team, animated conversation with someone, cross legs etc
A8: Miscellaneous
	Dance, jump, walk hand in hand with another (pretend) person, etc	
with 2 sets of clothing each. There is at least 1 clothing set per subject that is unique to that subject.

Each sequence is roughly 4 minutes, with each activity taking roughly 1 minute.

Each subject wears a different set of clothing in the two sequences. At least 1 set of clothing per subject is unique to that subject.

The dataset was recorded in a green screen studio with 14 cameras. The dataset has segmentation masks available for the background, for the chair, and for upper body and lower body clothing.
Use mpii_get_sequence_info to get information about which masks are available for each subject-sequence combination. The same function also provides information about the frame rate of the videos in the sequence, as well as the number of frames available per video.

The dataset is organized in the following hierarchy.
 SX: Where X is the subject ID (1 to 8)
	SeqY: Where Y is the sequence number (1 or 2)
		ChairMasks: Masks for the chair. This mask is encoded in the Red channel.
		FGmaks: Masks for the green screen, and the lower body and the upper body when available. See mpii_get_sequence_info for available augmentation opportunities. The green screen mask is in the Red channel, and when available, the green channel carries the upper body mask and the blue channel carries the lower body mask. It may be helpful to apply some gaussian smoothing to the masks when using them.
		imageSequence: The RGB frames.
		annot.mat: Body joint annotations in each camera's coordinate system. There are 2D pose annotations, 3D pose annotations and normalized 3D pose annotations (universal) available for each camera. For information about the joint order, joint labels and the joint subsets used in various projects, refer to mpii_get_joint_set. The file also contains the camera correspondence for each annotation cell (cameras, indexed with 0). For the camera subsets used in various projects, refer to mpii_get_camera_set. The 3D annotations (annot3) when reprojected into the image match the 2D annotations (annot2), however the same is not true of the normalized 3D annotations (univ_annot3). The 2D annotations (annot2) for each frame are arranged in a single row as x1,y1,x2,y2..,xj,yj, while the 3D annotations (annot3,univ_annot3) are arranged as x1,y1,z1,x2,y2,z2..,xj,yj,zj.. The file also contains the frame number correspondence for each row of annotations (frames). Though rare, it is possible that some sequences have a few frames missing at the end but annotations still available. Thus it is advisable to get the frame count F from mpii_get_sequence_info and read only the first F rows of annotations. 
		camera.calibration: Camera calibration parameters


The image frames of the dataset come in the form of video sequences, which are further grouped by common camera sets for ease of distribution. Before using the data, it is recommended to convert the videos back to image sequences using ffmpeg (ffmpeg -i "<some_folder>/video_X.avi" -qscale:v 1 "<some_folder>/img_X_%%06d.jpg") to ensure valid correspondence between the annotations and the frames.
