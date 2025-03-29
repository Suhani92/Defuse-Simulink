import numpy as np
import torch
import pandas as pd



def load_model(model_path=r"C:\Users\suhan\defuse\AI_ML_tools\ConvLSTM_pytorch\training\processed_model.pt"):

    model = torch.jit.load(r"C:\Users\suhan\defuse\AI_ML_tools\ConvLSTM_pytorch\training\processed_model.pt")
    model.eval()  
    return model

def predict(model,inputList):

    ip = np.array(inputList)
    print(ip.shape)

    input_tensor = torch.from_numpy(ip)
    #Permute the tensor to have the shape (batch size, channels, time)
    input_tensor = input_tensor.permute(0, 2, 1)  
    #forward pass
    output, _ = model(input_tensor)
    print(output)
    #Convert output back to a list and return to Simulink
    return output.squeeze(0).tolist()



