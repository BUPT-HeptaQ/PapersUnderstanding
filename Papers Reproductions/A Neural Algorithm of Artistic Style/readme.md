let cnn(X) be the network output with input image X

FXL ∈ cnn(X) Feature map as L-th layer of CNN

Tow kinds of loss functions:
content loss:
DLc (X, Y) = ||FXL – FYL||22  = ∑I  (FXL(i) – FYL(i))2
F: feature Map, C: content, X: output image, Y: input image

Style Loss:
DLs (X, Y) = ||GXL – GYL||22 = ∑k,l (GXL(k,l) – GYL(k,l))2
G: feature Map at L-th layer, it is the covariance matrix of input image X: output image, Y: input image, k: channel, l: l-th channel

where G is the correlation matrix between feature maps:
GXL(k,l) = <FkXL, FlXL> = ∑i(FkXL(i) – FlXL(i))2

Compute the gradient of the loss function w.r.t the input image to tune it
where w are the coefficients of different loss functions
