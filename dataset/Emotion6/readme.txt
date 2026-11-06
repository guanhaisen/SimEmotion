"Cornell Emotion6 Image Database"

The database is collected for academic research only. Image copyright remains with the original owners of the images.
If you use this database for research purposes, please cite us as follows in any resulting publication.

[1] Kuan-Chuan Peng, Amir Sadovnik, Andrew Gallagher, and Tsuhan Chen. "A Mixed Bag of Emotions: Model, Predict, and Transfer Emotion Distributions.", Computer Vision and Pattern Recognition (CVPR), 2015.


--------------------------------------------------------------------------
Description:

Placed under the folder "images," all the 1980 images in Emotion6 are collected from Flickr.
Each image in Emotion6 is placed under one of the six sub-folders under the folder "images."
There are 330 images under each sub-folder.
The name of the sub-folder is the emotion keyword used to search the images in that sub-folder.

The file "ground_truth.txt" lists the valence, arousal, and emotion distribution of each image in Emotion6.
The ground truth is collected by our user study mentioned in [1].
Each line in "ground_truth.txt" (except the first line) lists the information associated with one image.

The format is as follows:
[image_filename] [valence] [arousal] [prob. anger] [prob. disgust] [prob. fear] [prob. joy] [prob. sadness] [prob. surprise] [prob. neutral]

For example, the second line "disgust/1.jpg	2.5	3.8	0.133333	0.7	0	0.0666667	0.1	0	0" means that for the image placed in "images/disgust/1.jpg," the valence and arousal scores are 2.5 and 3.8 respectively.
The emotion distribution of the image "images/disgust/1.jpg" in terms of the probability of being classified in each emotion category is as follows:
the probability of being classified as anger is 0.133333;
the probability of being classified as disgust is 0.7;
the probability of being classified as fear is 0;
the probability of being classified as joy is 0.0666667;
the probability of being classified as sadness is 0.1;
the probability of being classified as surprise is 0;
the probability of being classified as neutral is 0.


--------------------------------------------------------------------------
Contact:

If you have any questions about the data, please contact Kuan-Chuan Peng (kp388@cornell.edu).
