tl.layers.Conv2d() parameters:

        n_filter (int) -- The number of filters.
        filter_size (tuple of int) -- The filter size (height, width).
        strides (tuple of int) -- The sliding window strides of corresponding input dimensions. 
                                  It must be in the same order as the shape parameter.
        dilation_rate (tuple of int) -- Specifying the dilation rate to use for dilated convolution.
        act (activation function) -- The activation function of this layer.
        padding (str) -- The padding algorithm type: "SAME" or "VALID".
        data_format (str) -- "channels_last" (NHWC, default) or "channels_first" (NCHW).
        W_init (initializer) -- The initializer for the the weight matrix.
        b_init (initializer or None) -- The initializer for the the bias vector. If None, skip biases.
        in_channels (int) -- The number of in channels.
        name (None or str) -- A unique layer name.
        
SubpixelConv2d() parameters:

        scale (int) -- The up-scaling ratio, a wrong setting will lead to dimension size error.
        n_out_channel (int or None) -- The number of output channels. 
            - If None, automatically set n_out_channel == the number of input channels / (scale x scale). 
            - The number of input channels == (scale x scale) x The number of output channels.
        act (activation function) -- The activation function of this layer.
        in_channels (int) -- The number of in channels.
        name (str) -- A unique layer name.
        
