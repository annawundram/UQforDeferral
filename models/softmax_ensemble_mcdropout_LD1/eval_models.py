# evaluation script for single Resnet50 ("Softmax"), Ensemble and Learned Deferral one-stage
from deferral_model import Model as def_Model
from torch.utils.data import DataLoader
from data.AIROGS_dataloader import AIROGS
from data.AIROGS_ood_dataloader import AIROGS_ood
import torch
import numpy as np
import gc

device = torch.device('cpu')

# load model and datasets
path = "/AIROGS.h5"
test_dataset = AIROGS(file_path=path, t="test", transform=None)

print(f"Test dataset length: {len(test_dataset)}")

test_dataloader = DataLoader(
        test_dataset, batch_size=4, shuffle=False, drop_last=False
)

# load ResNet50s with different seeds
model1 = def_Model.load_from_checkpoint("")
model2 = def_Model.load_from_checkpoint("")
model3 = def_Model.load_from_checkpoint("")
model4 = def_Model.load_from_checkpoint("")
model5 = def_Model.load_from_checkpoint("")
model6 = def_Model.load_from_checkpoint("")
model7 = def_Model.load_from_checkpoint("")
model8 = def_Model.load_from_checkpoint("")
model9 = def_Model.load_from_checkpoint("")
model10 = def_Model.load_from_checkpoint("")

# load Learned Deferral one-stage with different deferral costs
deferral_model0 = def_Model.load_from_checkpoint("") # c = 0.0
deferral_model1 = def_Model.load_from_checkpoint("") # c = 0.04
deferral_model2 = def_Model.load_from_checkpoint("") # c = 0.09
deferral_model3 = def_Model.load_from_checkpoint("") # c = 0.13
deferral_model4 = def_Model.load_from_checkpoint("") # c = 0.18
deferral_model5 = def_Model.load_from_checkpoint("") # c = 0.22
deferral_model6 = def_Model.load_from_checkpoint("") # c = 0.27
deferral_model7 = def_Model.load_from_checkpoint("") # c = 0.31
deferral_model8 = def_Model.load_from_checkpoint("") # c = 0.36
deferral_model9 = def_Model.load_from_checkpoint("") # c = 0.4

model1.eval()
model2.eval()
model3.eval()
model4.eval()
model5.eval()
model6.eval()
model7.eval()
model8.eval()
model9.eval()
model10.eval()

deferral_model0.eval()
deferral_model1.eval()
deferral_model2.eval()
deferral_model3.eval()
deferral_model4.eval()
deferral_model5.eval()
deferral_model6.eval()
deferral_model7.eval()
deferral_model8.eval()
deferral_model9.eval()

# global variables
ensemble_size = 10
n_deferral = 10
t = 15 # number of uncertainty thresholds
batch_size = 4

ensemble = []
single = []
deferral = []

for j, batch in enumerate(test_dataloader):
    x, y = batch
    x = x.to(device)

    out_ensemble = []

    with torch.no_grad():
        for i in range(1, ensemble_size + 1):
            model = globals().get(f"model{i}")
            model = model.to(device)

            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)

            # Move output to CPU immediately to free GPU memory
            out_cpu = out.cpu()

            if i == 1:
                single_out = out_cpu.numpy()
                if x.shape[0] < batch_size:
                    difference = batch_size - x.shape[0]
                    empty_array = np.zeros((difference, 2))
                    single_out = np.concatenate([single_out, empty_array], axis=0)
                single.append(single_out)

            out_ensemble.append(out_cpu.numpy())
            
            # Move model back to CPU and clean up
            model = model.to('cpu')
            del model

        # Convert ensemble outputs
        out_ensemble = np.stack(out_ensemble, axis=1)

        # Mean ensemble output
        mean_out_ensemble = out_ensemble.mean(axis=1)  # shape: (batch_size, 2)
        mean_out_ensemble = torch.from_numpy(mean_out_ensemble)

        if x.shape[0] < batch_size:
            difference = batch_size - x.shape[0]
            empty_array = np.zeros((difference, 10, 2))
            out_ensemble = np.concatenate([out_ensemble, empty_array], axis=0)
        ensemble.append(out_ensemble)

        # Deferral models
        list_deferral = []

        for i in range(4, 6):
            model = globals().get(f"deferral_model{i}")
            model = model.to(device)

            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)

            out_cpu = out.cpu()

            list_deferral.append(out_cpu.numpy())

            model = model.to('cpu')
            del model

        list_deferral = np.stack(list_deferral, axis=1)

        if x.shape[0] < batch_size:
            difference = batch_size - x.shape[0]
            empty_array = np.zeros((difference, n_deferral, 3))
            list_deferral = np.concatenate([list_deferral, empty_array], axis=0)
        deferral.append(list_deferral)

        # Clean up batch
        del x, y, out, out_cpu, list_deferral, mean_out_ensemble, out_ensemble
        gc.collect()
        torch.cuda.empty_cache()



# ---------- at epoch end ----------

ensemble = np.stack(ensemble)
single = np.stack(single)
deferral = np.stack(deferral)

# Sanity check

assert ensemble.shape == (len(test_dataloader), batch_size, ensemble_size, 2), f"Ensemble shape: {ensemble.shape}"
assert single.shape == (len(test_dataloader), batch_size, 2), f"Single shape: {single.shape}"
assert deferral.shape == (len(test_dataloader), batch_size, n_deferral, 3), f"Deferral shape: {deferral.shape}"

print("All shapes are correct!")

# save stuff
np.savez("model_outs", ensemble=ensemble, single=single, deferral=deferral)

# load noisy ood dataset
path = ""
ood_dataset = AIROGS_ood(file_path=path, transform=None)

print(f"Ood dataset length: {len(ood_dataset)}")

ood_dataloader = DataLoader(
        ood_dataset, batch_size=4, shuffle=False, drop_last=False
)

ensemble = []
single = []
deferral = []

for batch in iter(ood_dataloader):
    x, y = batch
    x = x.to(device)
    
    out_ensemble = []
    with torch.no_grad():
        
        for i in range(1, ensemble_size + 1):
            model = globals().get(f"model{i}")
            model = model.to(device)

            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)

            out_cpu = out.cpu()

            if i == 1:
                single_out = out_cpu.numpy()
                if x.shape[0] < batch_size:
                    difference = batch_size - x.shape[0]
                    empty_array = np.zeros((difference, 2))
                    single_out = np.concatenate([single_out, empty_array], axis=0)
                single.append(single_out)

            out_ensemble.append(out_cpu.numpy())

            # Move model back to CPU and clean up
            model = model.to('cpu')
            del model


        # all model outputs: shape (batch_size, ensemble_size, 2)
        out_ensemble = np.stack(out_ensemble, axis=1)
        if x.shape[0] < batch_size:
            difference = batch_size - x.shape[0]
            empty_array = np.zeros((difference, 10, 2))
            out_ensemble = np.concatenate([out_ensemble, empty_array], axis=0)
        ensemble.append(out_ensemble)
        
        list_deferral = []
        for i in range(0, 10):
            model = globals().get(f"deferral_model{i}")
            model = model.to(device)
            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)
            list_deferral.append(out.detach().cpu().numpy())

            model = model.to("cpu")
            del model

        list_deferral = np.stack(list_deferral, axis=1)
        if x.shape[0] < batch_size:
            difference = batch_size - x.shape[0]
            empty_array = np.zeros((difference, n_deferral, 3))
            list_deferral = np.concatenate([list_deferral, empty_array], axis=0)
        deferral.append(list_deferral)
    
    #clean up batch
    del x, y, out,list_deferral, out_cpu, out_ensemble
    gc.collect()


ensemble = np.stack(ensemble)
single = np.stack(single)
deferral = np.stack(deferral)

# Sanity check
assert ensemble.shape == (len(ood_dataloader), batch_size, ensemble_size, 2), f"Ensemble shape: {ensemble.shape}"
assert single.shape == (len(ood_dataloader), batch_size, 2), f"Single shape: {single.shape}"
assert deferral.shape == (len(ood_dataloader), batch_size, n_deferral, 3), f"Deferral shape: {deferral.shape}"

print("All shapes are correct!")
np.savez("model_outs_ood", ensemble=ensemble, single=single, deferral=deferral)

# load ood dataset
path = ""
ood_dataset = AIROGS_ood(file_path=path, transform=None)

print(f"Ood dataset length: {len(ood_dataset)}")

ood_dataloader = DataLoader(
        ood_dataset, batch_size=4, shuffle=False, drop_last=False
)

ensemble = []
single = []
deferral = []

for batch in iter(ood_dataloader):
    x, y = batch
    x = x.to(device)
    
    out_ensemble = []
    with torch.no_grad():
        
        for i in range(1, ensemble_size + 1):
            model = globals().get(f"model{i}")
            model = model.to(device)

            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)

            out_cpu = out.cpu()

            if i == 1:
                single_out = out_cpu.numpy()
                if x.shape[0] < batch_size:
                    difference = batch_size - x.shape[0]
                    empty_array = np.zeros((difference, 2))
                    single_out = np.concatenate([single_out, empty_array], axis=0)
                single.append(single_out)

            out_ensemble.append(out_cpu.numpy())

            # Move model back to CPU and clean up
            model = model.to('cpu')
            del model


        # all model outputs: shape (batch_size, ensemble_size, 2)
        out_ensemble = np.stack(out_ensemble, axis=1)
        if x.shape[0] < batch_size:
            difference = batch_size - x.shape[0]
            empty_array = np.zeros((difference, 10, 2))
            out_ensemble = np.concatenate([out_ensemble, empty_array], axis=0)
        ensemble.append(out_ensemble)
        
        list_deferral = []
        for i in range(0, 10):
            model = globals().get(f"deferral_model{i}")
            model = model.to(device)
            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)
            list_deferral.append(out.detach().cpu().numpy())

            model = model.to("cpu")
            del model

        list_deferral = np.stack(list_deferral, axis=1)
        if x.shape[0] < batch_size:
            difference = batch_size - x.shape[0]
            empty_array = np.zeros((difference, n_deferral, 3))
            list_deferral = np.concatenate([list_deferral, empty_array], axis=0)
        deferral.append(list_deferral)
    
    #clean up batch
    del x, y, out,list_deferral, out_cpu, out_ensemble
    gc.collect()


ensemble = np.stack(ensemble)
single = np.stack(single)
deferral = np.stack(deferral)

# Sanity check
assert ensemble.shape == (len(ood_dataloader), batch_size, ensemble_size, 2), f"Ensemble shape: {ensemble.shape}"
assert single.shape == (len(ood_dataloader), batch_size, 2), f"Single shape: {single.shape}"
assert deferral.shape == (len(ood_dataloader), batch_size, n_deferral, 3), f"Deferral shape: {deferral.shape}"

print("All shapes are correct!")
np.savez("model_outs_ood_blur", ensemble=ensemble, single=single, deferral=deferral)