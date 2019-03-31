# The data would be downloaded to this path
# Make sure you have approx 25GB space in this
# path to download the complete training set.
# The test set needs another 7GB and can be
# downloaded with get_testset.sh
destination='./'
# The subjects you want to download the train data for.
# Start with a few if all you want to do is examine the data
subjects=(5 6)
#subjects=(1 2 3 4 5 6 7 8)
# Set if you want to download the camera views
# that were not used for VNect
download_extra_wall_cameras=0
download_extra_ceiling_cameras=0
# Unset if you don't want to download the segmentation
# masks for the sequences
download_masks=0
# Set if you agree with the license conditions and want
# to proceed with downloading the dataset
ready_to_download=1
