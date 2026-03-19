import os, time, random
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import glob
from resnet import ResNet1d
from torch.utils.data import TensorDataset, random_split, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.distributed import init_process_group, destroy_process_group

########## set device (torchrun parallellization)
def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    init_process_group(backend="nccl")

######### set seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
##########

def compute_loss(ages, pred_ages, weights):
    diff = ages.flatten() - pred_ages.flatten()
    loss = torch.sum(weights.flatten()*diff*diff)
    return loss

def compute_weights(ages, max_weight=np.inf):
    _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    normalized_weights = weights / sum(weights)
    w = len(ages) * normalized_weights
    if max_weight < np.inf:
        w = np.minimum(w, max_weight)
        w = len(ages) * w / sum(w)
    return w

def _load_snap(model, gpu_id, snapshot_path):
    loc = f'cuda:{gpu_id}'
    snapshot = torch.load(snapshot_path, map_location=loc)
    model.module.load_state_dict(snapshot["MODEL_STATE"])
    print(f"=== Loaded an existing snapshot -- device: {gpu_id} ===")
    return snapshot["EPOCHS_RUN"]

def _save_snap(model, epoch, snapshot_path="snapshot.pt"):
    snapshot = {
        "MODEL_STATE": model.module.state_dict(),
        "EPOCHS_RUN": epoch
    }
    torch.save(snapshot, snapshot_path)
    print(f"Saved snapshot at Epoch: {epoch} === at {snapshot_path}")

# =========================================================================#
# ========================== Training Functions ===========================#
# =========================================================================#
def train_loop(epoch, chunk, dataloader, model, optimizer, device, snapshot_path="snapshot.pt"):
    if os.path.exists(snapshot_path):
        print("Loading snapshot...")
        epochs_run = _load_snap(model, device, snapshot_path)

    dataloader.sampler.set_epoch(epoch)
    # model to training mode (important to correctly handle dropout or batchnorm layers)
    model.train()

    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points
    # progress bar def
    train_pbar = tqdm(dataloader, desc="Training Epoch {epoch:2d} Chunk {chunk:2d}".format(epoch=epoch, chunk=chunk), leave=True)

    # training loop
    for i, (traces, diagnoses) in enumerate(train_pbar):
        traces, diagnoses = traces.to(device), diagnoses.to(device)
        
        # data to device (CPU or GPU if available)
        for j, (x,y) in enumerate(dataloader):
            x, y = x.transpose(1,2).to(device), y.to(device)
            yt, weights = y[:,0], y[:,1]
            pred = model(x)
            curr_loss = compute_loss(yt, pred, weights)
            curr_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            mae = mean_absolute_error(yt.detach().cpu(), pred.detach().cpu())
            mse = mean_squared_error(yt.detach().cpu(), pred.detach().cpu())

        # Update accumulated values
        total_loss += curr_loss.detach().cpu().numpy()
        n_entries += len(traces)

        # Update progress bar
        train_pbar.set_postfix({'loss': total_loss / n_entries})
    train_pbar.close()
    return total_loss / n_entries, mae, mse

def eval_loop(epoch, chunk, dataloader, model, device):
    # model to evaluation mode (important to correctly handle dropout or batchnorm layers)
    model.eval()
    
    # allocation
    total_loss = 0  # accumulated loss
    n_entries = 0   # accumulated number of data points

    # progress bar def
    eval_pbar = tqdm(dataloader, desc="Evaluation Epoch {epoch:2d} Chunk {chunk:2d}".format(epoch=epoch, chunk=chunk), leave=True)
    # evaluation loop
    for traces_cpu, diagnoses_cpu in eval_pbar:
        # data to device (CPU or GPU if available)
        traces, diagnoses = traces_cpu.to(device), diagnoses_cpu.to(device)
        with torch.no_grad():
            for x,y in dataloader:
                xt, y = x.transpose(1,2).to(device), y.to(device)
                yt, weights = y[:,0], y[:,1]
                pred = model(xt)
                curr_loss = compute_loss(yt, pred, weights)
 
                mae = mean_absolute_error(yt.detach().cpu(), pred.detach().cpu())
                mse = mean_squared_error(yt.detach().cpu(), pred.detach().cpu())

            # Update accumulated values
            total_loss += curr_loss.detach().cpu().numpy()
            n_entries += len(traces)
            # Update progress bar
            eval_pbar.set_postfix({
                'valid_loss': total_loss / n_entries, 
                'valid_mae': mae, 
                'valid_mse': mse,
                'baseline_mae': mean_absolute_error(yt.detach().cpu().numpy(), np.ones(len(yt))*torch.mean(yt).detach().cpu().numpy()), # if guessed the mean age as the default prediction
            })
    eval_pbar.close()
    return total_loss / n_entries, mae, mse
##########

# =========================================================================#
# ========================= Training PROCEDURE ============================#
# =========================================================================#

def main(num_rounds, num_chunks, id_=int(random.uniform(127962, 236777))):
    ddp_setup()
    gpu_id = int(os.environ["LOCAL_RANK"])
    # =============== Define model ============================================#
    tqdm.write("Define model...")
    model = ResNet1d(input_dim=(12, 4096),
                         blocks_dim=list(zip([64, 128, 196, 256, 320], # net_filter_size
                             [4096, 1024, 256, 64, 16])), # net_sequence_length
                         n_classes=1,
                         kernel_size=17,
                         dropout_rate=0.5)
    # model = CustomCNN()
    tqdm.write("Done!\n")
    
    learning_rate = 1e-3
    #weight_decay = 0.1
    num_epochs = num_rounds # 10
    batch_size = 256

    # =============== Define optimizer ========================================#
    tqdm.write("Define optimiser...")
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        #weight_decay=weight_decay
    )
    tqdm.write("Done!\n")
    
    # =============== Define lr scheduler =====================================#
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", patience=2)
   
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(gpu_id)
    model = DDP(model, device_ids=[gpu_id])

    # =============== Build data loaders ======================================#
    tqdm.write("Building data loaders...")
    
    vloaders = []
    for i, filepath in enumerate(sorted(glob.glob("../1-starter-ecg-model/data/code15-12l/*.hdf5"))):
        # build data loaders
        if filepath.replace("../1-starter-ecg-model/data/code15-12l/", "") in ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5"]:
            path_to_h5_train, path_to_csv_train = filepath, 'data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
        
            # load traces
            f = h5py.File(path_to_h5_train, 'r')
            traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
            
            # load labels
            ids_traces = np.array(f['exam_id'])
            df = pd.read_csv(path_to_csv_train).drop_duplicates(subset=["patient_id"])
            f.close()
            df.set_index('exam_id', inplace=True)
            df = df.reindex(ids_traces).dropna(subset=["age"]) # make sure the order is the same
            weights = compute_weights(df['age'])
            ages = torch.tensor(np.hstack((df['age'], weights), dtype=np.float16), dtype=torch.float16, device=gpu_id).reshape(-1,2)
            traces = traces[:ages.shape[0],:,:]
    
            # load dataset
            dataset = TensorDataset(traces, ages)
            vloaders.append(dataset)
            print("at", filepath, " >> put in validation!")

    vset = torch.utils.data.ConcatDataset(vloaders)
    vloader = DataLoader(vset, batch_size=512, shuffle=False, sampler=DistributedSampler(vset))

    # =============== Train model =============================================#
    tqdm.write("Training...")
    best_loss = np.inf

    # allocation
    train_loss_all, valid_loss_all = [], []
    maes, mses = [], []
    lrs = []
    counter = 0

    size = len(glob.glob("../1-starter-ecg-model/data/code15-12l/*.hdf5")) if num_chunks == -1 else num_chunks
    # loop over epochs
    for epoch in tqdm(range(1, num_epochs + 1)):
        for i, filepath in enumerate(sorted(glob.glob("../1-starter-ecg-model/data/code15-12l/*.hdf5"))[:size]):
            # build data loaders
            if filepath.replace("data/code15-12l/", "") not in ["exams_part0.hdf5", "exams_part1.hdf5", "exams_part2.hdf5", "exams_part3.hdf5"]:
                path_to_h5_train, path_to_csv_train = filepath, 'data/code15-12l/exams.csv' # path_to_records = 'data/codesubset/RECORDS.txt'
            
                # load traces
                f = h5py.File(path_to_h5_train, 'r')
                traces = torch.tensor(np.array(f['tracings'], dtype=np.float32), dtype=torch.float32)[:-1,:,:]
                
                # load labels
                ids_traces = np.array(f['exam_id'])
                df = pd.read_csv(path_to_csv_train).drop_duplicates(subset=["patient_id"])
                f.close()
                df.set_index('exam_id', inplace=True)
                df = df.reindex(ids_traces).dropna(subset=["age"]) # make sure the order is the same
                weights = compute_weights(df['age'])
                ages = torch.tensor(np.hstack((df['age'], weights), dtype=np.float16), dtype=torch.float16, device=gpu_id).reshape(-1,2)
                traces = traces[:ages.shape[0],:,:]
        
                # load dataset
                dataset = TensorDataset(traces, ages)
                tloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, sampler=DistributedSampler(dataset, shuffle=True))
            
                # training loop
                train_loss, trainmae, trainmse = train_loop(epoch, i, tloader, model, optimizer, device=gpu_id)
    
                # validation loop
                valid_loss, mae, mse = eval_loop(epoch, i, vloader, model, device=gpu_id)

                # collect losses
                train_loss_all.append(train_loss)
                valid_loss_all.append(valid_loss)
                
                # collect validation metrics
                maes.append(mae)
                mses.append(mse)
    
                # save checkpoints between epochs
                if gpu_id == 0:
                    _save_snap(model, epoch)
                    pd.DataFrame({
                        "epoch": np.arange(counter+1), 
                        "train_loss": train_loss_all, 
                        "valid_loss": valid_loss_all, 
                        "mae": maes, 
                        "mse": mses
                    }).to_csv(f"{id_}-results-partwise-lr{learning_rate}-ep{num_epochs}-exams{size-4}.csv", index=False)
     
                    # =============== PLOTTING  =============================================#
                    fig2 = plt.figure(figsize=(6,4), dpi=300)
                    ax2 = fig2.add_subplot()
                    ax2.set_title("Train-Validation Loss Curves - CODE-15% Centralized Training")
                    ax2.set_xlabel("Iterations")
                    ax2.set_ylabel("Loss")
                    ax2.plot(np.arange(counter+1)/(size-4), train_loss_all, color="blue", label="Train")
                    ax2.plot(np.arange(counter+1)/(size-4), valid_loss_all, color="orange", label="Validation")
                    ax2.legend(loc="best")
                    fig2.tight_layout()
                    fig2.savefig(f"{id_}-losses-partwise-age_mortalitypred-code15.png")
                    plt.close()
        
                    fig = plt.figure(figsize=(6,4), dpi=300)
                    ax = fig.add_subplot()
                    ax.set_title("MSE and MAE - CODE-15% Centralized Training")
                    ax.set_xlabel("Iterations")
                    ax.set_ylabel("Error")
                    ax.plot(np.arange(counter+1)/(size-4), maes, color="blue", label="MAE")
                    ax.plot(np.arange(counter+1)/(size-4), mses, color="orange", label="MSE")
                    ax.legend(loc="best")
                    fig.tight_layout()
                    fig.savefig(f"{id_}-maes-mses-partwise-age_mortalitypred-code15.png") 
                    plt.close()

                counter += 1

                # save best model: here we save the model only for the lowest validation loss
                if valid_loss < best_loss and gpu_id == 0:
                    # Save model parameters
                    torch.save({'model': model.state_dict()}, f'{id_}-resnetmodel-centralizedcode15-partwise.pth')
                    # Update best validation loss
                    best_loss = valid_loss
                
                # statement
                model_save_state = "best model -> saved" if valid_loss < best_loss else ""
                
                # Update learning rate with lr-scheduler
                if lr_scheduler:
                    print("lr >>>>", optimizer.param_groups[0]['lr'])
                    lrs.append(optimizer.param_groups[0]['lr'])
                    
                    # Print message
                    tqdm.write(f'******** Epoch {epoch}: Train Loss {train_loss} Valid Loss {valid_loss} --- {model_save_state} ********')

        if lr_scheduler and gpu_id == 0:
            fig2 = plt.figure(figsize=(6,4), dpi=300)
            ax2 = fig2.add_subplot()
            ax2.set_title("LR Scheduling - Centralized CODE-15%")
            ax2.set_xlabel("Iterations")
            ax2.set_ylabel("Learning Rate")
            ax2.plot(np.arange(counter)/(size-4), lrs, color="blue")
            ax2.grid(True)
            fig2.tight_layout()
            fig2.savefig(f"{id_}-lrscheduling-age_mortalitypred-code15.png")
            plt.close()
            lr_scheduler.step(valid_loss)
        if lrs[-1] < 1e-7:
            destroy_process_group()
            sys.exit(0)
                    
    destroy_process_group()
    # =======================================================================#

if __name__ == "__main__":
    main(num_rounds=8, num_chunks=-1, id_=int(random.uniform(1209310, 2230240)))
