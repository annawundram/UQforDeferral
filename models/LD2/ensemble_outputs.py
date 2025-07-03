# cpmputing and storing ensemble outputs + entropies as input to the mlp (second stage oin Liu et al method)

from softmax_ensemble_mcdropout_LD1.deferral_model import def_Model
from torch.utils.data import DataLoader
from data.AIROGS_dataloader import AIROGS
import torch
import numpy as np
import gc

device = torch.device('cpu')

# load model and datasets
path = "/AIROGS.h5"
train_dataset = AIROGS(file_path=path, t="train", transform=None)

print(f"Train dataset length: {len(train_dataset)}")

train_dataloader = DataLoader(
        train_dataset, batch_size=4, shuffle=False, drop_last=False
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

# global variables
ensemble_size = 10
batch_size = 4



ensemble = []
input = []
for j, batch in enumerate(train_dataloader):
    x, y = batch
    x = x.to(device)

    out_ensemble = []
    # first get ensemble outputs form frozen ensemble
    with torch.no_grad():
        for i in range(1, ensemble_size + 1):
            model = globals().get(f"model{i}")
            model = model.to(device)

            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)

            # Move output to CPU immediately to free GPU memory
            out_cpu = out.cpu()

            out_ensemble.append(out_cpu.numpy())
            
            # Move model back to CPU and clean up
            del model

        # Convert ensemble outputs
        out_ensemble = np.stack(out_ensemble, axis=1)
    out_ensemble = torch.from_numpy(out_ensemble)  # 4 x 10 x 2

    batch_inputs = []
    for i in range(out_ensemble.shape[0]):  # loop over batch of 4
        member = out_ensemble[i]  # shape: [10, 2]

        # Ensemble entropy
        probs = member.clamp(min=1e-8)
        entropy_e = -torch.sum(probs * torch.log(probs))  # scalar

        # Diagnostic entropy
        pred = torch.argmax(member, dim=1)
        counts = torch.bincount(pred, minlength=2).float()
        fractions = counts / counts.sum()
        fractions = fractions.clamp(min=1e-8)
        entropy_d = -torch.sum(fractions * torch.log(fractions))  # scalar

        # Flatten + append both entropies
        flattened = member.flatten()
        final_input = torch.cat([flattened, entropy_e.unsqueeze(0), entropy_d.unsqueeze(0)], dim=0)
        batch_inputs.append(final_input)

    # Final tensor: [4, 22] then flatten
    final_batch_input = torch.stack(batch_inputs, dim=0)
    final_batch_input = final_batch_input.to(device)
    final_batch_input = final_batch_input.flatten()

    
    input.append(final_batch_input)

    # Clean up batch
    del x, y, out, out_cpu, out_ensemble, batch_inputs
    gc.collect()
    torch.cuda.empty_cache()



# ---------- at epoch end ----------
input_train = torch.cat(input, dim=0)
input_train = input_train.reshape(-1, 22)
input_train = input_train.cpu().numpy()


# Sanity check

assert input_train.shape == (len(train_dataset), 22), f"Output shape: {input_train.shape}"

print("All shapes are correct!")


# load model and datasets
val_dataset = AIROGS(file_path=path, t="val", transform=None)

print(f"Val dataset length: {len(val_dataset)}")

val_dataloader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, drop_last=False
)

ensemble = []
input = []
for j, batch in enumerate(val_dataloader):
    x, y = batch
    x = x.to(device)

    out_ensemble = []
    # first get ensemble outputs form frozen ensemble
    with torch.no_grad():
        for i in range(1, ensemble_size + 1):
            model = globals().get(f"model{i}")
            model = model.to("cpu")

            out = model(x)
            out = torch.nn.functional.softmax(out, dim=1)

            # Move output to CPU immediately to free GPU memory
            out_cpu = out.cpu()

            out_ensemble.append(out_cpu.numpy())
            
            # Move model back to CPU and clean up
            del model

        # Convert ensemble outputs
        out_ensemble = np.stack(out_ensemble, axis=1)
    out_ensemble = torch.from_numpy(out_ensemble)  # 4 x 10 x 2

    batch_inputs = []
    for i in range(out_ensemble.shape[0]):  # loop over batch of 4
        member = out_ensemble[i]  # shape: [10, 2]

        # Ensemble entropy
        probs = member.clamp(min=1e-8)
        entropy_e = -torch.sum(probs * torch.log(probs))  # scalar

        # Diagnostic entropy
        pred = torch.argmax(member, dim=1)
        counts = torch.bincount(pred, minlength=2).float()
        fractions = counts / counts.sum()
        fractions = fractions.clamp(min=1e-8)
        entropy_d = -torch.sum(fractions * torch.log(fractions))  # scalar

        # Flatten + append both entropies
        flattened = member.flatten()
        final_input = torch.cat([flattened, entropy_e.unsqueeze(0), entropy_d.unsqueeze(0)], dim=0)
        batch_inputs.append(final_input)

    # Final tensor: [4, 22] then flatten
    final_batch_input = torch.stack(batch_inputs, dim=0)
    final_batch_input = final_batch_input.to(device)
    final_batch_input = final_batch_input.flatten()

    
    input.append(final_batch_input)

    # Clean up batch
    del x, y, out, out_cpu, out_ensemble, batch_inputs
    gc.collect()
    torch.cuda.empty_cache()

# ---------- at epoch end ----------
input_val = torch.cat(input, dim=0)
input_val = input_val.reshape(-1, 22)
input_val = input_val.cpu().numpy()


# Sanity check
assert input_val.shape == (len(val_dataset), 22), f"Output shape: {input_val.shape}"

print("All shapes are correct!")

# save stuff
np.savez("LD2_input", train=input_train, val=input_val)