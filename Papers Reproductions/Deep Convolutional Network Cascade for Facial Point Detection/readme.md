
Train the convolutional neural network to regress the feature point coordinates. 
If only one network is used for regression training, 
it will be found that the coordinates of the obtained feature points are not accurate enough, and cascade regression is adopted. 
The CNN method performs segmented feature point positioning to locate face feature points more quickly and accurately. 
If a larger network is used, the prediction of feature points will be more accurate and robust, but the time will increase; 
in order to find a balance between speed and performance, use a smaller network, so use the idea of cascading, first Rough detection, 
then fine-tune feature points


(1) Firstly, a network is trained on the entire face image (red box) to perform coarse regression on the coordinates of the face feature points. 
The actual adopted network has a face area grayscale map with an input size of 39*39.
The approximate position of the feature points can be obtained; as shown in level 1 above, in the green box, 5 points are predicted; 
the first layer is divided into three waves, which are five points, left and right eyes and nose, nose and mouth.

(2) Design another regression network to train the local area image (the yellow area in level 2 and level 3) around the face feature point 
(obtaining the feature point obtained after level 1 training) as the input, 
and the actual network used is Enter a grayscale map of the feature point local area with a size of 15*15 
to predict a more accurate feature point position. Here level3 is smaller than the input area defined by level2.

Another thing to note is that the Euclidean loss used in the regression, when calculating the coordinates, 
uses the relative coordinates instead of the absolute coordinates, that is, each coordinate calculation, 
the relative coordinates are relative to the yellow box boundary shown in the above figure. 
The absolute coordinates are based on the green border boundary.

In addition, during the level1 training, the training set was also augmented. 
In addition to mirroring, two sets of scaling and four sets of panning are performed on the face frame, 
as well as two sets of small angle rotations, and then the face frame area is cropped into a 39*39 size area.

Deep CNN F1 convolution network structure, the input layer of the level1 network uses a 39*39 single-channel gray image, 
passes through four convolution layers with pooling layers, and finally passes through the fully connected layer, 
outputting a dimension of 10 As a result, the coordinate values representing the five feature points, 
and the last layer is the Euclidean loss layer, and the calculation is the accumulation of the mean error 
between the network predicted coordinate value and the true value (both relative values).

