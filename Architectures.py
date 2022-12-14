"""
Contains the model architectures so that they may easily be called upon.
"""
from distutils.command.sdist import sdist
import inspect
from sklearn.ensemble import RandomForestRegressor

# Some mess here that could be sorted out
from tensorflow import keras
from keras import backend as K, Model, Input, optimizers, layers
from keras.layers import Dense, Dropout, Conv1D, Conv2D, Layer, BatchNormalization, LayerNormalization
from keras.layers import Activation, SpatialDropout1D, SpatialDropout2D, Lambda, Flatten
from tensorflow_addons.layers import WeightNormalization
from numpy import array
from keras.utils.vis_utils import plot_model
from Seismic_interp_ToolBox import ai_to_reflectivity, reflectivity_to_ai

class ResidualBlock(Layer):
    """
    If one would wish to write this as a class. Inspired by keras-tcn
    """
    def __init__(self,
                 dilation_rate: int,
                 nb_filters: int,
                 kernel_size,
                 padding: str,
                 activation: str = 'relu',
                 convolution_type: str = 'Conv2D',
                 dropout_type: str ='spatial',
                 dropout_rate: float = 0.,
                 kernel_initializer: str = 'he_normal',
                 use_batch_norm: bool = False,
                 use_layer_norm: bool = False,
                 use_weight_norm: bool = False,
                 **kwargs): # Any initializers for the Layer class
        """
        docstring here %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        """
        if convolution_type == 'Conv2D':
            self.dilation_rate = (1, dilation_rate) # Height, width
        else:
            self.dilation_rate = dilation_rate
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.activation = activation
        self.convolution_type = convolution_type # My addition
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.kernel_initializer = kernel_initializer
        self.layers = []
        self.shape_match_conv = None
        self.res_output_shape = None
        self.final_activation = None

        super(ResidualBlock, self).__init__(**kwargs)
    
    def compute_output_shape(self, input_shape):
        """
        Inspired by a function of the same name in another TCN implementation
        Not sure why input_shape is not used, but required as an input.
        """
        return [self.res_output_shape, self.res_output_shape]

    def _build_layer(self, layer):
        """Helper function for building layer
        Args:
            layer: Appends layer to internal layer list and builds it based on the current output
                   shape of ResidualBlocK. Updates current output shape.
        """
        self.layers.append(layer)
        self.layers[-1].build(self.res_output_shape)
        self.res_output_shape = self.layers[-1].compute_output_shape(self.res_output_shape) # Not sure if compute output shape does anything here

    def build(self, input_shape):
        if self.convolution_type.lower() == 'conv1d': c_func = Conv1D
        else: c_func = Conv2D

        with K.name_scope(self.name):
            self.layers = []
            self.res_output_shape = input_shape

            for k in range(2):
                name = f'{self.convolution_type.lower()}_{k}'
                with K.name_scope(name):
                    conv = c_func(
                        filters=self.nb_filters,
                        kernel_size=self.kernel_size,
                        dilation_rate=self.dilation_rate,
                        padding=self.padding,
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
                    if self.use_weight_norm:
                        from tensorflow_addons.layers import WeightNormalization
                        # WeightNormalization API is different than other Normalizations; requires wrapping
                        with K.name_scope('norm_{}'.format(k)):
                            conv = WeightNormalization(conv)
                    self._build_layer(conv)

                # Other Normalization types
                with K.name_scope('norm_{}'.format(k)):
                    if self.use_batch_norm:
                        self._build_layer(BatchNormalization())
                    elif self.use_layer_norm:
                        self._build_layer(LayerNormalization())
                    elif self.use_weight_norm:
                        pass # Already done above
                
                with K.name_scope('act_and_dropout_{}'.format(k)):
                    if self.convolution_type.lower() == 'conv1d':
                        dim = 1
                    elif self.convolution_type.lower() == 'conv2d':
                        dim = 2
                    if self.dropout_type == 'normal':
                        d_func = Dropout
                        dname = 'Dropout'
                    if self.dropout_type == 'spatial':
                        dname = 'SDropout'
                        if dim == 1:
                            d_func = SpatialDropout1D
                        elif dim == 2:
                            d_func = SpatialDropout2D

                    self._build_layer(Activation(self.activation, name='Act_{}_{}'.format(self.convolution_type, k)))
                    self._build_layer(d_func(rate=self.dropout_rate, name='{}{}D_{}'.format(dname, dim, k)))
    
            if self.nb_filters != input_shape[-1]:
                # 1x1 convolution mathes the shapes (channel dimension).
                name = 'matching_conv1D'
                with K.name_scope(name):

                    self.shape_match_conv = c_func(
                        filters=self.nb_filters,
                        kernel_size=1,
                        padding='same',
                        name=name,
                        kernel_initializer=self.kernel_initializer
                    )
            else:
                name = 'matching_identity'
                self.shape_match_conv = Lambda(lambda x: x, name=name)
            
            with K.name_scope(name):
                self.shape_match_conv.build(input_shape)
                self.res_output_shape = self.shape_match_conv.compute_output_shape(input_shape)
            
            self._build_layer(Activation(self.activation, name='Act_Conv_Blocks'))
            self.final_activation = Activation(self.activation, name='Act_Res_Block')
            self.final_activation.build(self.res_output_shape) # According to philipperemy this probably is not be necessary

            # Forcing keras to add layers in the list to self._layers
            for layer in self.layers:
                self.__setattr__(layer.name, layer)
            self.__setattr__(self.shape_match_conv.name, self.shape_match_conv)
            self.__setattr__(self.final_activation.name, self.final_activation)

            super(ResidualBlock, self).build(input_shape) # This to make sure self.built is set to True

    def call(self, inputs, training=None, **kwargs):
        x1 = inputs
        for layer in self.layers:
            training_flag = 'traning' in dict(inspect.signature(layer.call).parameters)
            x1 = layer(x1, training=training) if training_flag else layer(x1)
        x2 = self.shape_match_conv(inputs)
        x1_x2 = self.final_activation(layers.add([x2, x1], name='Add_Res'))
        return [x1_x2, x1]


class TCN(Layer):
    """
    Creates a TCN layer.
    """

    def __init__(self,
                 nb_filters=64,
                 kernel_size=(3, 9),
                 nb_stacks=1,
                 dilations=(1, 2, 4, 8, 16, 32),
                 padding='causal',
                 use_skip_connections=True,
                 dropout_type = 'spatial',
                 dropout_rate=0.0,
                 return_sequences=False,
                 activation='relu',
                 convolution_type = 'Conv2D',
                 kernel_initializer='he_normal',
                 use_batch_norm=False,
                 use_layer_norm=False,
                 use_weight_norm=False,
                 **kwargs):
        
        self.return_sequences = return_sequences
        self.dropout_type = dropout_type
        self.dropout_rate = dropout_rate
        self.use_skip_connections = use_skip_connections
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters
        self.activation_name = activation
        self.convolution_type = convolution_type
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm
        self.skip_connections = []
        self.residual_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None
        self.slicer_layer = None  # in case return_sequence=False
        self.output_slice_index = None  # in case return_sequence=False
        self.padding_same_and_time_dim_unknown = False  # edge case if padding='same' and time_dim = None

        if self.use_batch_norm + self.use_layer_norm + self.use_weight_norm > 1:
            raise ValueError('Only one normalization can be specified at once.')
        
        if isinstance(self.nb_filters, list):
            assert len(self.nb_filters) == len(self.dilations)
            if len(set(self.nb_filters)) > 1 and self.use_skip_connections:
                raise ValueError('Skip connections are not compatible'
                                 'with a list of filters, unless they are all equal.')
        if padding != 'causal' and padding != 'same':
            raise ValueError('Only \'causal\' or \'same\' padding are compatible for this layer.')
        
        # Initialize parent class (..which is Layer?)
        super(TCN, self).__init__(**kwargs)
    
    @property
    def receptive_field(self):
        return 1 + 2*(self.kernel_size-1)*self.nb_stacks*sum(self.dilations) # May need to pick the kernel dimension

    def build(self, input_shape):

        self.build_output_shape = input_shape

        self.residual_blocks = []
        total_num_blocks = self.nb_stacks * len(self.dilations)
        if not self.use_skip_connections:
            total_num_blocks += 1 # A cheap way to do a false case for below
    
        for s in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                if isinstance(self.nb_filters, list):
                    res_block_filters = self.nb_filters[i] 
                else:
                    res_block_filters = self.nb_filters
                self.residual_blocks.append(ResidualBlock(dilation_rate=d,
                                                          nb_filters=res_block_filters,
                                                          kernel_size=self.kernel_size,
                                                          padding=self.padding,
                                                          activation=self.activation_name,
                                                          convolution_type=self.convolution_type,
                                                          dropout_type=self.dropout_type,
                                                          dropout_rate=self.dropout_rate,
                                                          use_batch_norm=self.use_batch_norm,
                                                          use_layer_norm=self.use_layer_norm,
                                                          use_weight_norm=self.use_weight_norm,
                                                          kernel_initializer=self.kernel_initializer,
                                                          name='residual_block_{}'.format(len(self.residual_blocks))))

        # this is done to force keras to add the layers in the list to self._layers
        for layer in self.residual_blocks:
            self.__setattr__(layer.name, layer)

        self.output_slice_index = None
        if self.padding == 'same':
            time = self.build_output_shape.as_list()[1]
            if time is not None:
                self.output_slice_index = int(self.build_output_shape.as_list()[1] / 2)
            else:
                # It will be known at call time
                self.padding_same_and_time_dim_unknown = True
        else:
            self.output_slice_index = -1 # causal case.
        self.slicer_layer = Lambda(lambda tt: tt[:, self.output_slice_index, :], name='Slice_Output')
        self.slicer_layer.build(self.build_output_shape.as_list())

    # Not needed function that Philippe Remy wrote
    def compute_output_shape(self, input_shape):
        """
        Overridden in case keras uses it somewhere... no idea. Just trying to avoid future errors.
        """
        if not self.built:
            self.build(input_shape)
        if not self.return_sequences:
            batch_size = self.build_output_shape[0]
            batch_size = batch_size.value if hasattr(batch_size, 'value') else batch_size
            nb_filters = self.build_output_shape[-1]
            return [batch_size, nb_filters]
        else:
            # Compatibility tensorflow 1.x
            return [v.value if hasattr(v, 'value') else v for v in self.build_output_shape]

    def call(self, inputs, training=None, **kwargs):
        x = inputs
        self.layers_outputs = [x]
        self.skip_connections = []
        for res_block in self.residual_blocks:
            try:
                x, skip_out = res_block(x, training=training)
            except TypeError: # backwards compatibility
                x, skip_out = res_block(K.cast(x, 'float32'), training=training)
            self.skip_connections.append(skip_out)
            self.layers_outputs.append(x)
        
        if self.use_skip_connections:
            x = layers.add(self.skip_connections, name='Add_Skip_Connections')
            self.layers_outputs.append(x)
        
        if not self.return_sequences:
            # Case: time dimension is unknown. e.g. (bs, None, input_dim).
            if self.padding_same_and_time_dim_unknown:
                self.output_slice_index = K.shape(self.layers_outputs[-1])[1] // 2
            x = self.slicer_layer(x)
            self.layers_outputs.append(x)
        return x
    def get_config(self):
        """
        Returns the config of a the layer. This is used for saving and loading from a model
        :return: python dictionary with specs to rebuild layer
        """
        config = super(TCN, self).get_config() # Non-recursive, uses Layer.get_config(); key names must be standardized
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['dilations'] = self.dilations
        config['padding'] = self.padding
        config['use_skip_connections'] = self.use_skip_connections
        config['dropout_type'] = self.dropout_type
        config['dropout_rate'] = self.dropout_rate
        config['return_sequences'] = self.return_sequences
        config['activation'] = self.activation_name
        config['convolution_type'] = self.convolution_type # May need another name
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        config['kernel_initializer'] = self.kernel_initializer
        return config


class CNN(Layer):

    def __init__(self,
                nb_filters=64,
                kernel_size=3,
                nb_stacks=3,
                padding='same',
                activation='relu',
                convolution_type = 'Conv2D',
                kernel_initializer='he_normal',
                dropout_rate = 0.001,
                dilations = [1, 2, 4, 8],
                use_dropout = False,
                use_batch_norm=False,
                use_layer_norm=False,
                use_weight_norm=False,
                **kwargs):
        
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.nb_stacks = nb_stacks
        self.padding = padding
        self.dilations = dilations
        self.activation = activation
        self.convolution_type = convolution_type
        self.kernel_initializer = kernel_initializer
        self.dropout_rate = dropout_rate
        self.use_dropout = use_dropout


        # Not sure if needed..
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_norm = use_weight_norm

        self.conv_blocks = []
        self.layers_outputs = []
        self.build_output_shape = None

        self.conv_func = Conv2D
        if convolution_type == 'Conv1D':
            self.conv_func = Conv1D

        super(CNN, self).__init__(**kwargs)

        
        

    def build(self, input_shape):

        self.build_output_shape = input_shape
        self.conv_blocks = []

        for k in range(self.nb_stacks):
            for i, d in enumerate(self.dilations):
                conv_filters = self.nb_filters[i] if isinstance(self.nb_filters, list) else self.nb_filters
                self.conv_blocks.append(self.conv_func(filters=conv_filters, 
                                                    kernel_size=self.kernel_size,
                                                    padding = self.padding,
                                                    activation=self.activation,
                                                    kernel_initializer=self.kernel_initializer,
                                                    dilation_rate=d,
                                                    name='convolution_layer_{}'.format(len(self.conv_blocks))))
        
        for layer in self.conv_blocks:
            self.__setattr__(layer.name, layer)


    def call(self, inputs, training=None, **kwargs):
        x = inputs
        self.layers_outputs = [x]
        for conv_block in self.conv_blocks:
            try:
                x = conv_block(x, training=training)
            except TypeError: # also backwards compatibiltiy
                x = conv_block(K.cast(x, 'float32'), training=training)
                self.layers_outputs.append(x)
        return x

    def get_config(self):
        config = super(CNN, self).get_config()
        config['nb_filters'] = self.nb_filters
        config['kernel_size'] = self.kernel_size
        config['nb_stacks'] = self.nb_stacks
        config['padding'] = self.padding
        config['activation'] = self.activation
        config['convolution_type'] = self.convolution_type
        config['kernel_initializer'] = self.kernel_initializer
        config['use_batch_norm'] = self.use_batch_norm
        config['use_layer_norm'] = self.use_layer_norm
        config['use_weight_norm'] = self.use_weight_norm
        return config


def compiled_TCN(training_data, config, **kwargs):
    """
    @ Author: Sjur [in progress]
    Three temporal blocks as feature extractions

    Split into three for regression, and three for reconstruction

   This function only works for reconstruction at present moment 
    """
    nb_filters              = config['nb_filters']
    kernel_size             = config['kernel_size']
    nb_tcn_stacks           = config['nb_tcn_stacks']
    nb_reg_stacks           = config['nb_reg_stacks']
    nb_rec_stacks           = config['nb_rec_stacks']
    dilations               = config['dilations']
    padding                 = config['padding']
    use_skip_connections    = config['use_skip_connections']
    dropout_type            = config['dropout_type']
    dropout_rate            = config['dropout_rate']
    return_sequences        = config['return_sequences']
    activation              = config['activation']
    convolution_type        = config['convolution_type']
    lr                      = config['learn_rate']
    kernel_initializer      = config['kernel_initializer']
    use_batch_norm          = config['use_batch_norm']
    use_layer_norm          = config['use_layer_norm']
    use_weight_norm         = config['use_weight_norm']

    batch_size              = config['batch_size']
    epochs                  = config['epochs']


    # Data
    X, y = training_data

    input_shape = tuple([*X.shape[1:], 1])
    input_layer = Input(shape=input_shape)

    # Feature Extraction module
    x = TCN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_tcn_stacks,
            dilations=dilations,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_type = dropout_type,
            dropout_rate=dropout_rate,
            return_sequences=return_sequences,
            activation=activation,
            convolution_type=convolution_type,
            kernel_initializer=kernel_initializer,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
            name='Feature_recognition_module'
    )(input_layer)
    
    # Regression module
    reg = CNN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_reg_stacks,
            padding='same',
            activation='tanh',
            convolution_type=convolution_type,
            kernel_initializer=kernel_initializer,
            name = 'Regression_module'
            )(x)
    
    c_func = Conv1D
    if convolution_type == 'Conv2D': c_func = Conv2D # Not quite sure    

    reg = c_func(1, kernel_size, padding=padding, activation='linear', name='regression_output')(reg)

    # Reconstruciton module
    rec = CNN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_rec_stacks,
            padding=padding,
            activation=activation,
            convolution_type=convolution_type,
            kernel_initializer=kernel_initializer,
            name = 'Reconstruction_module'
            )(x)


    rec = c_func(1, kernel_size, padding=padding, activation='tanh', name='reconstruction_output')(rec)

    output_layer = [reg, rec] # Regression, reconstruction

    gen_model = Model(inputs = input_layer, 
                      outputs = output_layer)
    

    seis_disc_model = discriminator(output_layer[1].shape, 3, name='seismic_discriminator')
    ai_disc_model = discriminator(output_layer[0].shape, 3, name='ai_discriminator')


    model = multi_task_GAN([ai_disc_model, seis_disc_model], gen_model)

    model.compile(keras.optimizers.Adam(lr=lr, clipnorm=1.), loss={'regression_output' : 'mean_squared_error',
                                                                           'reconstruction_output' : 'mean_squared_error'})
    

    print(model.summary())

    History = model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, **kwargs)
    
    return model, History


def unsupervised_Marmousi(train_data, config, **kwargs):
    nb_filters              = config['nb_filters']
    kernel_size             = config['kernel_size']
    nb_tcn_stacks           = config['nb_tcn_stacks']
    nb_reg_stacks           = config['nb_reg_stacks']
    dilations               = config['dilations']
    padding                 = config['padding']
    use_skip_connections    = config['use_skip_connections']
    dropout_type            = config['dropout_type']
    dropout_rate            = config['dropout_rate']
    return_sequences        = config['return_sequences']
    activation              = config['activation']
    convolution_type        = config['convolution_type']
    lr                      = config['learn_rate']
    kernel_initializer      = config['kernel_initializer']
    use_batch_norm          = config['use_batch_norm']
    use_layer_norm          = config['use_layer_norm']
    use_weight_norm         = config['use_weight_norm']

    batch_size              = config['batch_size']
    epochs                  = config['epochs']


    # Data
    X = train_data

    input_shape = tuple([*X.shape[1:], 1])
    input_layer = Input(shape=input_shape)

    # Feature Extraction module
    x = TCN(nb_filters=nb_filters,
            kernel_size=kernel_size,
            nb_stacks=nb_tcn_stacks,
            dilations=dilations,
            padding=padding,
            use_skip_connections=use_skip_connections,
            dropout_type = dropout_type,
            dropout_rate=dropout_rate,
            return_sequences=return_sequences,
            activation=activation,
            convolution_type=convolution_type,
            kernel_initializer=kernel_initializer,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_norm=use_weight_norm,
            name='Feature_recognition_module'
    )(input_layer)

    # Regression module
    reg = CNN(nb_filters=nb_filters,
              kernel_size=3,
              nb_stacks=nb_reg_stacks,
              padding=padding,
              activation='sigmoid',
              convolution_type=convolution_type,
              kernel_initializer=kernel_initializer,
              dilations=dilations,
              name = 'Regression_module'
              )(x)
    
    c_func = Conv1D
    if convolution_type == 'Conv2D': c_func = Conv2D # Not quite sure    

    reg = c_func(1, kernel_size=3, padding=padding, activation='linear', name='regression_output')(reg)

    # reg = Flatten()(x)
    # reg = Dense(X.shape[1]//2, activation='sigmoid')(reg)
    # reg = Dense(X.shape[1], activation='linear')(reg)
    

    output_layer = [reg] # Regression, reconstruction

    model = Model(inputs = input_layer, 
                  outputs = output_layer)
    model.compile(keras.optimizers.Adam(lr=lr, clipnorm=1.), loss=conv_refl_loss)

    print(model.summary())

    History = model.fit(x=X, y=X, batch_size=batch_size, epochs=epochs, **kwargs)
    
    return model, History


def discriminator(Input_shape, depth, conv_dim=1, name='discriminator'):
    input_layer = Input(Input_shape)
    x = input_layer
    for _ in range(depth):
        x = Conv1D(1, kernel_size=4, padding='valid')(x)
        x = layers.BatchNormalization(scale=False)(x)
        x = layers.LeakyReLU()(x)
    x = layers.Flatten()(x)
    output_score = Dense(1)(x)
    return Model(input_layer, output_score, name=name)


# Loss Function
import numpy as np
import tensorflow as tf
from keras.losses import mean_squared_error
import tensorflow as tf
# from keras.backend import conv1d
from tensorflow.nn import convolution

w = np.load('Marmousi_data/wavelet_Marmousi.npz')['wavelet']
#wavelet = tf.convert_to_tensor(w, tf.float32)
wavelet = tf.constant(w, shape=[len(w), 1, 1, 1], dtype=tf.float32)

def conv_ai_loss(y_true, y_pred):
    """
    takes the predicted acoustic impedance, and convolves it to see if it matches the seismic
    """
    y_refl = ai_to_reflectivity(y_pred)
    shp = tf.shape(y_refl)
    y_refl = tf.reshape(y_refl, (shp[0], shp[1], 1, 1))
    y_pred_seis = convolution(y_refl, wavelet, padding='VALID')
    print(y_pred_seis.shape)
    y_pred_seis = tf.reshape(y_pred_seis, (shp[0], shp[1]))
    return mean_squared_error(y_true, y_pred_seis)


def conv_refl_loss(y_true, y_pred):
    """
    takes the predicted acoustic impedance, and convolves it to see if it matches the seismic
    """
    shp = tf.shape(y_pred)
    y_refl = tf.reshape(y_pred, (shp[0], shp[1], 1, 1))
    y_pred_seis = convolution(y_refl, wavelet, padding='SAME')
    y_pred_seis = tf.reshape(y_pred_seis, (shp[0], shp[1]))
    return mean_squared_error(y_true, y_pred_seis)



def ai_to_reflectivity(ai):
    '''
    Acoustic Impedance to Reflectivity
    '''    
    refl = (ai[:, 1:]-ai[:, :-1])/(ai[:, 1:]+ai[:, :-1])
    padding = tf.constant([[0, 0], [1, 0]])
    return tf.pad(refl, padding)


# Models:
def create_discriminator():
    img_input = Input()



class multi_task_GAN(Model):

    def __init__(self, discriminators, generator):
        """
        """
        super(multi_task_GAN, self).__init__()
        self.seismic_discriminator  = discriminators[1]
        self.ai_discriminator       = discriminators[0]
        self.generator              = generator
        self.gen_seis_loss          = keras.metrics.Mean(name='generator_seismic_loss')
        self.gen_ai_loss            = keras.metrics.Mean(name='generator_ai_loss')
        self.disc_seis_loss         = keras.metrics.Mean(name='discriminator_seismic_loss')
        self.disc_ai_loss           = keras.metrics.Mean(name='discriminator_ai_loss')
    
    @property
    def metrics(self):
        pass

    def compile(self, g_optimizer, d_optimizers, g_loss, d_loss):
        super(multi_task_GAN, self).compile()
        self.g_optimizer    = g_optimizer
        self.d_X_optimizer  = d_optimizers[1]
        self.d_y_optimizer  = d_optimizers[0]
        self.g_loss         = g_loss
        self.d_loss         = d_loss
    
    def train_step(self, batch_data):
        real_X, real_y = batch_data
        with tf.GradientTape(persistent=True) as tape:
            fake_y, fake_X = self.generator(real_X, training=True)
            disc_real_X = self.seismic_discriminator(real_X, training=True)
            disc_fake_X = self.seismic_discriminator(fake_X, training=True)
            disc_real_y = self.ai_discriminator(real_y, training=True)
            disc_fake_y = self.ai_discriminator(fake_y, training=True)

            # Generator loss
            gen_X_loss = self.gen_seis_loss(disc_fake_X)
            gen_y_loss = self.gen_ai_loss(disc_fake_y)

            # Discriminator loss
            disc_X_loss = self.disc_seis_loss(disc_real_X, disc_fake_X)
            disc_y_loss = self.disc_ai_loss(disc_real_y, disc_fake_y)

        print('\nGenerator\n')
        print(self.generator.trainable_variables)
        print('\nDiscriminator\n')
        print(self.seismic_discriminator.trainable_variables)

        # Get the gradients
        gen_y_grads, gen_X_grads = tape.gradient([gen_y_loss, gen_X_loss], self.generator.trainable_variables)
        disc_X_grads = tape.gradient(disc_X_loss, self.seismic_discriminator.trainable_variables)
        disc_y_grads = tape.gradient(disc_y_loss, self.ai_discriminator)

        # Update the weights
        self.g_optimizer.apply_gradients(
            zip(gen_X_grads, self.generator.trainable_variables)
        )
        self.d_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.seismic_discriminator.trainable_variables)
        )
        self.d_y_optimizer.apply_gradients(
            zip(disc_y_grads, self.ai_discriminator.trainable_variables)
        )

        return {
                'gen_X_loss' : gen_X_loss,
                'gen_y_loss' : gen_y_loss,
                'disc_X_loss': disc_X_loss,
                'disc_y_loss': disc_y_loss
                }