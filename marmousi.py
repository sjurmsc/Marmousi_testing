from Architectures import *
from Feat_aug import *
import matplotlib.pyplot as plt
from Seismic_interp_ToolBox import ai_to_reflectivity
import numpy as np
# import segyio

files = {'Marmousi' : {'density'  : 'Marmousi_data/density_marmousi-ii.segy',
                       'velocity' : 'Marmousi_data/vp_marmousi-ii.segy'},
         'SEAM' : {'density'      : 'SEAM_data/SEAM_Den_Elastic_N23900.sgy',
                   'velocity'     : 'SEAM_data/SEAM_Vp_Elastic_N23900.sgy'}}

depth = 10000

dataset = 'SEAM'

density, rho_z =    get_traces(files[dataset]['density'], zrange=(None, depth))
velocity, v_z =     get_traces(files[dataset]['velocity'], zrange=(None, depth))

ai = density*velocity
print(ai)

w = np.load('Marmousi_data/wavelet_Marmousi.npz')
wavelet = w['wavelet']
dt = w['dt']

seis = []
slopes = []
for i, trace in enumerate(ai):
    print(i)
    refl, slope = ai_to_reflectivity(trace)
    s = np.convolve(refl, wavelet, mode='same')
    seis.append(s)
    slopes.append(slope)

seis = np.array(seis)
slopes = np.array(slopes)

# seis = np.load('Data_dumps/seis1.npy')


config = dict()
config['nb_filters']            = [3, 3, 5, 5, 5, 5, 5]
config['kernel_size']           = 7 # Height, width
config['dilations']             = [1, 2, 4, 8, 16, 32, 64]
config['padding']               = 'same'
config['use_skip_connections']  = False
config['dropout_type']          = 'normal'
config['dropout_rate']          = 0.03
config['return_sequences']      = True
config['activation']            = 'relu'
config['convolution_type']      = 'Conv1D'
config['learn_rate']            = 0.001
config['kernel_initializer']    = 'he_normal'

config['use_batch_norm']        = False
config['use_layer_norm']        = False
config['use_weight_norm']       = True

config['nb_tcn_stacks']         = 3
config['nb_reg_stacks']         = 5
config['nb_rec_stacks']         = 3  

config['batch_size']            = 20
config['epochs']                = 8

split = len(seis)//2
train = [seis[:split], ai[:split]]
test =  [seis[split:], ai[split:]]

model, History = compiled_TCN(train, config=config)
#model.save('model')
pred = model.predict(test)
#pred = pred.reshape(pred.shape[:-1])
np.save('Data_dumps/pred', pred)
#plt.imshow(pred[0].T, cmap='Spectral')
#plt.show()




