
import h5py

def load_data(dataset):
    data = h5py.File(dataset, 'r')
    sequence_code = data['sequences'].value
    label = data['labs'].value
    return [sequence_code, label]



def get_data(data_path):
    data_path = data_path

    training_dataset = data_path + "train.hdf5"

    X,Y= load_data(training_dataset)

    return  X, Y

