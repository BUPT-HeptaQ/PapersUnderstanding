background data set:  http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

Basic Functions:

1. tf.nn.moments(original_input, axes=[0, 1, 2])
# average and standard deviation of the first three dimensions, the result is the last dimension,
that is, the average and standard deviation of each feature_map

Parameter description: original_input is the input feature_map, axes=[0, 1, 2] averages the three dimensions,
that is, each feature_map gets an average and standard deviation

2. tf.control_dependencies([train_mean, train_var]):
When executing the operation inside with, it will execute train_mean and train_var first.

Parameter description: train_mean indicates the assignment to pop_mean,
and train_var indicates the assignment to pop_var.

3. tf.cond(is_training, batch_normalize_train, batch_normalize_inference)
# If is_training is true, execute the batch_normalize_train function operation.
If it is false, execute batch_normalize_inference operation.

Parameter description: is_training is true and false, batch_normalize_train indicates the normalize of the training,
and batch_normalize_inference indicates the normalize of the test.

4. tf.nn.atrous_conv2d(original_input, filters, dilated, padding)
# Perform a hole convolution operation to increase the receptive field of convolution

Parameter description: original_input indicates the input sample, filter indicates the convolution kernel,
dialed indicates the number of zeros of the convolution kernel, and padding indicates that the feature is zero-filled.

5. tf.nn.conv2d_transpose(original_input, filters, output_size, strides)
# Perform deconvolution

Parameter description: original_input represents the input sample, filter represents the convolution kernel,
output_size represents the dimension of the output, strides represents the multiple of the image expansion

6.tf.nn.batch_normalization(original_input, mean, var, beta, scale, episilon)
# normalize operation

Parameter description: original_input represents the input sample, mean represents the average of the convolution,
var represents the standard deviation of the convolution, beta represents the deviation,
scale represents the normalized range, and episilon prevents the denominator from being 0.

7.tf.train.get_checkpoint_state('./backup')
# Determine whether there is a checkpoint in the file used for model saving, that is, whether it has been saved before.

Parameter description: './backup' file name saved by sess



The following is the structure of the paper, mainly consisting of two structures.

The first structure is a fully-convolutional neural network:

first use a mask and a picture to combine to form a picture with a vacancy,
input the vacant picture into the full convolution, after stride is equal to 2, do two downward convolutions Then,
after 4 different_conv (cavity convolution), in order to increase the field of view of the convolution kernel without
losing the dimension, finally use the inverse convolution of zeros to increase the dimension,
and then pass the last two layers of convolution. Formed a picture.

The second structure is global_discriminator and local_discriminator:

The size of the global_image input is the size of the full image,
and the input size of local_image is half of the full image size.
The size of the mask in the above figure is 96-128 and inside the local_image, the value range is a random value.

After global_image and local_image are convoluted, the output of 1024 is output.
We will concatenate the output tf.concat and predict the final output [1] to determine whether the picture is the
actual picture (true) or the filled picture (false).


Code: network.py: Build the Network class, construct the generated network, generate self.completion,
construct the discriminant network, get the original_output, local_image output self.real,
get the global_completion, local_completion output self.fake,
use l2_loss construct to generate the network The loss value generator_loss,
using the cross entropy structure to discriminate the loss value of the network discriminator_loss

Step 1: use the self.generator construct to generate the network, the input is x*(1-mask), is_training

Step 2: Construct self.completion using self.imitation * mask + x * (1-mask)

Step 3: Use self.discriminator, enter original_input, local_image, is_training, reuse=False,
for real sample discrimination self.real

Step 4: Use self.discriminator, input as global_complement, local_complement, is_training, reuse=True,
used to generate samples for self.fake

Step 5: Use tf.nn.l2_loss to generate the loss value of mse,
which is used to generate the loss value of the network, self.generator_loss

Step 6: Use tf.reduce_mean(tf.softmax..logits(labels=tf.ones_like(real), logits=real)) to obtain the true cross entropy.
 Similarly, obtain the cross entropy of the generated samples, and add the two.
 And multiply the coefficient to get the final loss value, self.discriminator_loss.

Step 7: Use tf.get_collection(tf.GraphKey.TRAINABLE_VARIABLES, scope=‘generator’)
and scope = ‘discriminator’ to get the parameters for generating the network and discriminating the network.

