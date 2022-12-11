"""
Functions to be used for feature augmentation
"""
import segyio
import json
from numpy import array, row_stack, intersect1d, where, amax, amin
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from os import listdir
from pathlib import Path
import sys
# from JR.Seismic_interp_ToolBox import ai_to_reflectivity, reflectivity_to_ai

# Functions for loading data

def get_data_loc():
    d_filepath = './Data/data.json'
    return json.loads(d_filepath)


def get_traces(fp, mmap=True, zrange: tuple = (None,100)):
    """
    This function should conserve some information about the domain (time or depth) of
    the data.
    """
    with segyio.open(fp, ignore_geometry=True) as seis_data:
        z = seis_data.samples
        if mmap:
            seis_data.mmap()  # Only to be used if the file size is small compared to available memory
        traces = segyio.collect(seis_data.trace)
    
    traces, z = traces[:, z<zrange[1]], z[z<zrange[1]]
    return traces, z


def get_matching_traces(fp_X, fp_y, mmap = True, zrange: tuple = (25, 100), group_traces: int = 1, trunc = False):
    """
    Function to assist in maintaining cardinality of the dataset
    %%%%%%%%%%%%%%%% Can add overlap here
    """
    assert group_traces%2, 'Amount of traces must be odd to have a center trace'

    with segyio.open(fp_X, ignore_geometry=True) as X_data:
        with segyio.open(fp_y, ignore_geometry=True) as y_data:
            # retrieving depth values for target and input data
            z_X = X_data.samples
            z_y = y_data.samples

            # getting index of max depth for truncation
            X_max_idx = amax(where(z_X < zrange[1])) + 1
            y_max_idx = amax(where(z_y < zrange[1])) + 1

            # The acoustic impedance starts at depth 25m
            X_min_idx = amin(where(z_X >= z_y[0]))


            if mmap: X_data.mmap(); y_data.mmap() # initiate mmap mode for large datasets

            # get information about what traces are overlapping
            nums_X = segyio.collect(X_data.attributes(segyio.TraceField.CDP))
            nums_y = segyio.collect(y_data.attributes(segyio.TraceField.CDP))
            CDP, idx_X, idx_y = intersect1d(nums_X, nums_y, return_indices=True)
            assert len(idx_X) == len(idx_y)

            # collect the data
            X_traces = segyio.collect(X_data.trace)[idx_X, X_min_idx:X_max_idx]
            y_traces = segyio.collect(y_data.trace)[idx_y, :y_max_idx]

            # Using JR code to resample the acoustic impedance
 
            # y_refl, slope = ai_to_reflectivity(y_traces)
            # y_interp = interp1d(z_y[:y_max_idx], y_refl, kind='nearest', axis=1)
            # y_interp_refl = array(y_interp(z_X[X_min_idx:X_max_idx]))
            # y_traces = reflectivity_to_ai(y_interp_refl, slope)

    if not group_traces == 1:
        num_traces = X_traces.shape[0]
        len_traces = X_traces.shape[1]
        num_images = num_traces//group_traces
        indices_truncated = num_images*group_traces
        discarded_images = num_traces-indices_truncated
        l_indices = (discarded_images//2); r_indices = indices_truncated + l_indices + discarded_images%2
        X_traces = X_traces[l_indices:r_indices].reshape((num_images, group_traces, len_traces))
        y_traces = y_traces[l_indices:r_indices].reshape((num_images, group_traces, len_traces))
    
    if trunc:  # Done as a quick way to remove bad data, as it is most often at the ends
        X_traces = X_traces[trunc:-trunc]
        y_traces = y_traces[trunc:-trunc]
    
    return X_traces, y_traces, (z_X, z_y)


def sgy_to_keras_dataset(X_data_label_list,
                         y_data_label_list,
                         test_size=0.2, 
                         group_traces = 1,
                         zrange: tuple = (None, 100), 
                         reconstruction = True,
                         validation = False, 
                         X_normalize = 'StandardScaler',
                         y_normalize = 'MinMaxScaler',
                         random_state=1,
                         shuffle=True,
                         fraction_data=False,
                         truncate_data=False):
    """
    random_state may be passed for recreating results
    """
    data_dict = load_data_dict()

    # Something to evaluate that z is same for all in a feature
    X = array([]); y = array([])

    for i, key in enumerate(X_data_label_list):
        x_dir = Path(data_dict[key])
        y_dir = Path(data_dict[y_data_label_list[i]])
        matched = match_files(x_dir, y_dir)
        if fraction_data: matched = matched[:int(len(matched)*fraction_data)] # %%%%%%%%%%%%%%%%%%%%%quickfix
        m_len = len(matched)
        for i, (xm, ym) in enumerate(matched):
            # Giving feedback to how the collection is going
            sys.stdout.write('\rCollecting trace data into dataset {}/{}'.format(i+1, m_len))
            sys.stdout.flush()

            try: x_traces, y_traces, z = get_matching_traces(xm, ym, zrange=zrange, group_traces=group_traces, trunc=truncate_data)
            except: print('\nCould not load file {}\n'.format(xm)); continue

            if not len(X):
                X = array(x_traces)
                y = array(y_traces)
            else:
                X = row_stack((X, x_traces))
                y = row_stack((y, y_traces))
    sys.stdout.write('\n'); sys.stdout.flush()
    
    X_scaler = None
    y_scaler = None
    # Normalization
    if X_normalize == 'MinMaxScaler':
        X_scaler = MinMaxScaler()
        X_new = X_scaler.fit_transform(X, y)
        X = X_new
    elif X_normalize == 'StandardScaler':
        X_scaler = StandardScaler()
        X_new = X_scaler.fit_transform(X, y)
        X = X_new
    if y_normalize == 'MinMaxScaler':
        y_scaler = MinMaxScaler()
        y_new = y_scaler.fit_transform(y)
        y = y_new


    train_X, test_X, train_y, test_y = train_test_split(X, y, 
                                                        test_size=test_size, 
                                                        random_state=random_state,
                                                        shuffle=shuffle)  # dataset must be np.array
    
    if validation:
        test_X, val_X, test_y, val_y = train_test_split(test_X, test_y, test_size=test_size, random_state=random_state, shuffle=shuffle)
        if reconstruction:
            train_y = [train_y, train_X]
            test_y = [test_y, test_X]
            val_y = [val_y, val_X]
        return (train_X, train_y), (test_X, test_y), (val_X, val_y)
    
    if reconstruction:
        train_y = [train_y, train_X]
        test_y = [test_y, test_X]
    return (train_X, train_y), (test_X, test_y), (X_scaler, y_scaler)


def match_files(X_folder_loc, y_folder_loc, file_extension='.sgy'):
    """
    Matches the features from two seperate traces by filename
    Filenames corresponding to each other are returned as a list
    of tuples in the form [(X_file, y_file)]
    """
    X_dir = list(Path(X_folder_loc).glob('*'+file_extension)); y_dir = list(Path(y_folder_loc).glob('*'+file_extension))

    file_pairs = []
    for i, fname in enumerate(X_dir):
        X_name = fname.name[:fname.name.find('_MIG_DPT.sgy')]
        j_list = []
        for j, yfile in enumerate(y_dir):
            y_name = yfile.name[:yfile.name.find('_MIG.Abs_Zp.sgy')]
            if X_name == y_name:
                file_pairs.append((str(fname), str(yfile)))
                j_list.append(j)
        [y_dir.pop(j) for j in j_list]
    return file_pairs
          

def pair_well_and_seismic():
    """
    For a given log trace, this function locates the appropriate seismic trace,
    and allows for adding the neighboring traces to output a seismic image
    centered at the well position. 
    """
    pass


class CPT_TRACE:

    def __init__(self, name, coords, trace, z):
        self.name = name
        self.lon = coords[0]
        self.lat = coords[1]
        self.trace = trace
        self.z = z
    
    def closest_seismic(self, seis_coords, seis_CDP):
        """
        Given a input of a list of seismic coordinates on the form seis_coords[0, :] = longitudes, seis_coords[1, :] = latitudes
        this function returns the value which is closest to the trace in space

        The function does not consider coordinate transformations and assumes that the given coords are in the same
        coordinate system.
        """
        seis_coords = np.array(seis_coords)
        seis_CDP = np.array(seis_CDP)
        assert len(seis_coords[0])==len(seis_CDP)
        d = np.sqrt((self.lon-seis_coords[0, :])**2+(self.lat-seis_coords[1, :])**2)

        return seis_CDP[:, np.where(d==np.min(d))]





def load_data_dict():
    data_json = './Data/data.json'
    with open(data_json, 'r') as readfile:
        data_dict = json.loads(readfile.read())
    return data_dict


def update_data_dict():
    """ Edit this funciton to change the filepaths to the relevant data
        Filepaths must be relative to the repository, which is in user folder.
        Double dot (..) steps outside of this folder to access the OneDrive
        folder
    """
    data_json = './Data/data.json'
    root = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/'
    data_dict = {
        '00_AI'                 : root + '00_AI',
        '2DUHRS_06_MIG_DEPTH'   : root + '2DUHRS_06_MIG_DEPTH'
    }
    with open(data_json, 'w') as writefile:
        writefile.write(json.dumps(data_dict, indent=2))


def combine_seismic_traces():
    pass


def find_nth(haystack, needle, n : int):
    n = abs(n); assert n > 0
    max_needle_amount = len(haystack.split(needle)); assert n < max_needle_amount
    if n-1:
        intermed = haystack.find(needle) + 1
        loc = intermed + find_nth(haystack[intermed:], needle, n-1)
    else:
        return haystack.find(needle)
    return loc


def find_duplicates(m_files):
    dupes = dict()
    names = []
    for i, (X_m, y_m) in enumerate(m_files):

        name = X_m[62:-12]
        name_list = name.split('_')
        for n in names:
            n_list = n.split('_')
            if (len(name_list)>3) and (len(n_list)>3):
                if n_list[:3] == name_list[:3]:
                    key = '_'.join(n_list[:3])
                    if not (key in dupes.keys()):
                        dupes[key] = [X_m[:62] + '_'.join(n_list) + X_m[-12:], X_m]
                    else:
                        dupes[key].append(X_m)
        names.append(name)
    return dupes


def box_plots_for_dupelicates():
    import matplotlib.pyplot as plt
    import numpy as np

    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    lines = []
    dupe_names = []
    dupe_labels = []
    dupe = find_duplicates(m_files=m_files)
    for key, val in dupe.items():
        trc = []
        lbls = []
        for file in val:
            trace, _ = get_traces(file, zrange=(25, 100))
            trc.append(trace.flatten())
            lbls.append(file[62:-12])
        lines.append(trc)
        dupe_names.append(key)
        dupe_labels.append(lbls)
    
    for name, labels, collection in zip(dupe_names, dupe_labels, lines):
        plt.clf()
        plt.boxplot(collection, labels=labels)
        plt.title(name)
        plt.xticks(rotation=10)
        plt.savefig('Data/dupelicates/{}.png'.format(name))


def img_plots_for_dupelicates():
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    lines = []
    dupe_names = []
    dupe_labels = []
    dupe = find_duplicates(m_files=m_files)
    for key, val in dupe.items():
        trc = []
        lbls = []
        for file in val:
            trace, _ = get_traces(file, zrange=(25, 100))
            trc.append(trace.T)
            lbls.append(file[62:-12])
        lines.append(trc)
        dupe_names.append(key)
        dupe_labels.append(lbls)
    
    for name, labels, collection in zip(dupe_names, dupe_labels, lines):
        plt.clf()
        fig, ax = plt.subplots(len(labels))
        fig.tight_layout(h_pad=1)
        for i, im in enumerate(collection):
            norm = Normalize(-2, 2)
            ax[i].imshow(im, cmap = 'seismic', norm=norm)
            ax[i].set_title(label=labels[i], fontsize=10)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
        plt.savefig('Data/dupelicates/Image_{}.png'.format(name))


def negative_ai_values():
    import matplotlib.pyplot as plt
    import numpy as np
    from random import randint
    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    low_val = 0
    # y_below_zero = []
    t_b_z = []
    for file, i in m_files:
        # _, z = get_traces(file, zrange=(25, 100))
        traces, z_ai = get_traces(i, zrange=(25, 100))
        t = traces[np.where(np.any((traces<low_val), axis=1)), :]
        t_b_z+=list(t[0])
        # y_below_zero.append(t_b_z)
    # plt.hist(y_below_zero)
    print(np.shape(t_b_z))
    r = randint(0, len(t_b_z))
    plt.plot(t_b_z[r], z_ai)
    print('Minimum on the plot is: {}'.format(np.min(t_b_z[r])))
    print('Total minimum is {}'.format(np.min(t_b_z)))
    plt.show()

# Only used for testing the code
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np
    from random import randint
    d_dict = load_data_dict()
    m_files = match_files(d_dict['2DUHRS_06_MIG_DEPTH'], d_dict['00_AI'])
    z_high = []
    y_below_zero = []
    t_b_z = []
    for file, i in m_files:
        # _, z = get_traces(file, zrange=(25, 100))
        traces, z_ai = get_traces(i, zrange=(25, 100))
        # r = randint(0, len(traces))
        # plt.plot(traces[r], z_ai[::-1])
        # flat_t = traces.flatten()
        t = traces[np.where(np.any((traces<-10000), axis=1)), :]
        # print(np.shape(t))
        t_b_z+=list(t[0])
        
        #y_below_zero.append(list(t_b_z))

    # plt.hist(y_below_zero)
    print(np.shape(t_b_z))
    r = randint(0, len(t_b_z))
    plt.plot(t_b_z[r], z_ai)
    print('Minimum on the plot is: {}'.format(np.min(t_b_z[r])))
    print('Total minimum is {}'.format(np.min(t_b_z)))
    plt.show()

    # plt.boxplot(lines)
    # plt.show()
    # sgy_to_keras_dataset(['2DUHRS_06_MIG_DEPTH'], ['00_AI'], fraction_data=0.05, group_traces=3, normalize='StandardScaler')
