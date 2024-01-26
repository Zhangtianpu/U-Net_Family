import os
import pickle
import torch

def save_logs(data,folder_path, filename="experiment_result.pkl"):
    if os.path.isdir(folder_path) is False:
        os.mkdir(folder_path)
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


def load_logs(folder_path, filename='experiment_result.pkl'):
    file_path = os.path.join(folder_path, filename)
    if os.path.isfile(file_path) is False:
        raise Exception("File doesn't exist")

    with open(file_path, 'rb') as f:
        data=pickle.load(f)
        return data

def load_model(model,experiment_dir,model_name,epoch,device):
    load_model_path=os.path.join(experiment_dir,'%s_%s.pkl'%(model_name,epoch))
    model.load_state_dict(torch.load(load_model_path,map_location=device))

