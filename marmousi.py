from Architectures import *
from Feat_aug import *
import matplotlib.pyplot as plt
from Seismic_interp_ToolBox import ai_to_reflectivity
import numpy as np
import sys
from tensorflow.keras.layers import Conv1D
# import segyio
files = dict()
# files['Marmousi'] =  {'density'  : 'Marmousi_data/density_marmousi-ii.segy', 'velocity' : 'Marmousi_data/vp_marmousi-ii.segy'}
files['SEAM']     =  {'density'      : 'SEAM_data/SEAM_Den_Elastic_N23900.sgy', 'velocity'     : 'SEAM_data/SEAM_Vp_Elastic_N23900.sgy'}

depth = 10_000

dataset = 'SEAM'

density, rho_z =    get_traces(files['SEAM']['density'],  zrange=(None, depth))
velocity, v_z =     get_traces(files['SEAM']['velocity'], zrange=(None, depth))

ai = density*velocity

# w = np.load('Marmousi_data/wavelet_Marmousi.npz')
# wavelet = w['wavelet']
# dt = w['dt']

# seis = []
# slopes = []
# for i, trace in enumerate(ai):
#     sys.stdout.write('\r' + str(i))
#     refl, slope = ai_to_reflectivity(trace)
#     s = np.convolve(refl, wavelet, mode='same')
#     seis.append(s)
#     slopes.append(slope)
# sys.stdout.flush()

# seis = np.array(seis)
# slopes = np.array(slopes)
# np.save('Data_dumps/SEAM_seismic.npy', seis)
seis = np.load('Data_dumps/SEAM_seismic.npy')

config = dict()
config['nb_filters']            = 8
config['kernel_size']           = (3, 9) # Height, width
config['dilations']             = [1, 2, 4, 8, 16, 32, 64]
config['padding']               = 'same'
config['use_skip_connections']  = True
config['dropout_type']          = 'normal'
config['dropout_rate']          = 0.01
config['return_sequences']      = True
config['activation']            = 'relu'

config['learn_rate']            = 0.001
config['kernel_initializer']    = 'he_normal'

config['use_adversaries']       = True
config['alpha']                 = 0.6
config['beta']                  = 0.4

config['use_batch_norm']        = False
config['use_layer_norm']        = False
config['use_weight_norm']       = True

config['nb_tcn_stacks']         = 3
config['nb_reg_stacks']         = 5
config['nb_rec_stacks']         = 3 

config['batch_size']            = 20
config['epochs']                = 100

config['group_traces']          = 7

config['convolution_func']      = Conv1D
if config['group_traces']>1: config['convolution_func'] = Conv2D
else: config['kernel_size'] = config['kernel_size'][1]

# if len(ai_datasets):
#         train_data, test_data, scalers = sgy_to_keras_dataset([files['SEAM']['density']], 
#                                                               [], 
#                                                               fraction_data=0.01, 
#                                                               test_size=0.8, 
#                                                               group_traces=group_traces, 
#                                                               X_normalize='StandardScaler',
#                                                               y_normalize='MinMaxScaler',
#                                                               shuffle=False,
#                                                               truncate_data=0)
#         test_X, test_y = test_data
n_traces, trace_len = ai.shape
r = np.floor(n_traces/config['group_traces']).astype(np.int)
n = r*config['group_traces']
ai = ai[:n, :].reshape((r, config['group_traces'], trace_len))
seis = seis[:n, :].reshape((r, config['group_traces'], trace_len))

split = len(seis)//2
train = [seis[:split], [ai[:split], seis[:split]]]
test =  seis[split:]

model, History = compiled_TCN(train, config=config)
#model.save('model')
pred = model.predict(test)
pred = np.array(pred)
ai_pred, seis_pred = pred

ai_pred = ai_pred.reshape(split*config['group_traces'], trace_len)
seis_pred = seis_pred.reshape(split*config['group_traces'], trace_len)
# np.save('Data_dumps/pred', pred)
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(ai_pred.T, cmap='Spectral')
axs[1, 0].imshow(ai[split:].reshape(split*config['group_traces'], trace_len).T, cmap='Spectral')
axs[0, 1].imshow(seis_pred.T, cmap='seismic')
axs[1, 1].imshow(seis[split:].reshape(split*config['group_traces'], trace_len).T, cmap='seismic')
plt.show()




