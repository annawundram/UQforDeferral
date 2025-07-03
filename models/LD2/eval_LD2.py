from two_stage_deferral import Deferral_Model
import torch
import numpy as np
import gc
import h5py

hdf5_file = h5py.File("/AIROGS.h5", "r")
labels = hdf5_file["test/diagnosis"][:]

device = torch.device('cpu')

liu_model0 = Deferral_Model.load_from_checkpoint("") # alpha = 0.4
liu_model1 = Deferral_Model.load_from_checkpoint("") # alpha = 0.47
liu_model2 = Deferral_Model.load_from_checkpoint("") # alpha = 0.55
liu_model3 = Deferral_Model.load_from_checkpoint("") # alpha = 0.62
liu_model4 = Deferral_Model.load_from_checkpoint("") # alpha = 0.69
liu_model5 = Deferral_Model.load_from_checkpoint("") # alpha = 0.76
liu_model6 = Deferral_Model.load_from_checkpoint("") # alpha = 0.84
liu_model7 = Deferral_Model.load_from_checkpoint("") # alpha = 0.91
liu_model8 = Deferral_Model.load_from_checkpoint("") # alpha = 0.99
liu_model9 = Deferral_Model.load_from_checkpoint("") # alpha = 1.06

liu_model0.eval()
liu_model1.eval()
liu_model2.eval()
liu_model3.eval()
liu_model4.eval()
liu_model5.eval()
liu_model6.eval()
liu_model7.eval()
liu_model8.eval()
liu_model9.eval()


batch_size = 4
n_deferral = 10
counter = 0

deferral_liu = []
# load ensemble outputs - only run after ensemble has been run
model_outs = np.load("/model_outs.npz")
ensemble = model_outs["ensemble"] # in last batch: last three entries are zero
    
# Deferral models Liu et al.
for j, batch in enumerate(ensemble):
    if j == ensemble.shape[0] - 1:
        x = batch[0]
        y = labels[-1]
        y = np.array([y])
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y).long()
        x = x.unsqueeze(0)

    else:
        x = batch
        y = labels[counter : counter + 4]
        x = torch.from_numpy(x.astype(np.float32))
        y = torch.from_numpy(y).long()
    

    batch_inputs = []
    for i in range(x.shape[0]):  # loop over batch of 4
        member = x[i]  # shape: [10, 2]

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

    # Final tensor: [4, 22]
    final_batch_input = torch.stack(batch_inputs, dim=0)
    final_batch_input = final_batch_input.to(device)

    list_deferral = []
    with torch.no_grad():
        for i in range(0, n_deferral):
            model = globals().get(f"liu_model{i}")
            model = model.to(device)

            out = model(final_batch_input)
            out = torch.nn.functional.softmax(out, dim=1)

            out_cpu = out.cpu()
            list_deferral.append(out_cpu.numpy())

            model = model.to('cpu')
            del model

        list_deferral = np.stack(list_deferral, axis=1)

        if out_cpu.shape[0] < batch_size:
            difference = batch_size - out_cpu.shape[0]
            empty_array = np.zeros((difference, n_deferral, 3))
            list_deferral = np.concatenate([list_deferral, empty_array], axis=0)
        deferral_liu.append(list_deferral)

        counter += 4
        # Clean up batch
        del x, out, out_cpu, list_deferral
        gc.collect()
        torch.cuda.empty_cache()

deferral_liu = np.stack(deferral_liu)

assert deferral_liu.shape == (ensemble.shape[0], batch_size, n_deferral, 3), f"Deferral shape: {deferral_liu.shape}"

np.savez("model_outs_liu", deferral_liu=deferral_liu)

# noise
deferral_liu = []
# load ensemble outputs - only run after ensemble has been run
model_outs = np.load("/model_outs_ood.npz")

ensemble = model_outs["ensemble"] # in last batch: last three entries are zero
    
# Deferral models Liu et al.
for j, batch in enumerate(ensemble):
    if j == ensemble.shape[0] - 1:
        x = batch[0]
        x = torch.from_numpy(x.astype(np.float32))
        x = x.unsqueeze(0)
    else:
        x = batch
        x = torch.from_numpy(x.astype(np.float32))

    batch_inputs = []
    for i in range(x.shape[0]):  # loop over batch of 4
        member = x[i]  # shape: [10, 2]

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

    # Final tensor: [4, 22]
    final_batch_input = torch.stack(batch_inputs, dim=0)
    final_batch_input = final_batch_input.to(device)

    list_deferral = []
    with torch.no_grad():
        for i in range(0, n_deferral):
            model = globals().get(f"liu_model{i}")
            model = model.to(device)

            print(final_batch_input.shape[0])

            out = model(final_batch_input)
            out = torch.nn.functional.softmax(out, dim=1)

            out_cpu = out.cpu()
            list_deferral.append(out_cpu.numpy())

            print(out_cpu.shape[0])

            model = model.to('cpu')
            del model

        list_deferral = np.stack(list_deferral, axis=1)

        if out_cpu.shape[0] < batch_size:
            difference = batch_size - out_cpu.shape[0]
            print(difference)
            empty_array = np.zeros((difference, n_deferral, 3))
            list_deferral = np.concatenate([list_deferral, empty_array], axis=0)
        deferral_liu.append(list_deferral)


        # Clean up batch
        del x, out, out_cpu, list_deferral
        gc.collect()
        torch.cuda.empty_cache()

deferral_liu = np.stack(deferral_liu)

assert deferral_liu.shape == (ensemble.shape[0], batch_size, n_deferral, 3), f"Deferral shape: {deferral_liu.shape}"

np.savez("model_outs_liu_ood", deferral_liu=deferral_liu)

# blur
deferral_liu = []
# load ensemble outputs - only run after ensemble has been run
model_outs = np.load("/model_outs_ood_blur.npz")

ensemble = model_outs["ensemble"] # in last batch: last three entries are zero
    
# Deferral models Liu et al.
for j, batch in enumerate(ensemble):
    if j == ensemble.shape[0] - 1:
        x = batch[0]
        x = torch.from_numpy(x.astype(np.float32))
        x = x.unsqueeze(0)
    else:
        x = batch
        x = torch.from_numpy(x.astype(np.float32))

    batch_inputs = []
    for i in range(x.shape[0]):  # loop over batch of 4
        member = x[i]  # shape: [10, 2]

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

    # Final tensor: [4, 22]
    final_batch_input = torch.stack(batch_inputs, dim=0)
    final_batch_input = final_batch_input.to(device)

    list_deferral = []
    with torch.no_grad():
        for i in range(0, n_deferral):
            model = globals().get(f"liu_model{i}")
            model = model.to(device)

            print(final_batch_input.shape[0])

            out = model(final_batch_input)
            out = torch.nn.functional.softmax(out, dim=1)

            out_cpu = out.cpu()
            list_deferral.append(out_cpu.numpy())

            print(out_cpu.shape[0])

            model = model.to('cpu')
            del model

        list_deferral = np.stack(list_deferral, axis=1)

        if out_cpu.shape[0] < batch_size:
            difference = batch_size - out_cpu.shape[0]
            print(difference)
            empty_array = np.zeros((difference, n_deferral, 3))
            list_deferral = np.concatenate([list_deferral, empty_array], axis=0)
        deferral_liu.append(list_deferral)


        # Clean up batch
        del x, out, out_cpu, list_deferral
        gc.collect()
        torch.cuda.empty_cache()

deferral_liu = np.stack(deferral_liu)

assert deferral_liu.shape == (ensemble.shape[0], batch_size, n_deferral, 3), f"Deferral shape: {deferral_liu.shape}"

np.savez("model_outs_liu_ood_blur", deferral_liu=deferral_liu)


