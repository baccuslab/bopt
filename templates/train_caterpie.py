import asimov as asmv
import wandb
import os
import scipy
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import datetime

# Clear GPU memory
torch.cuda.empty_cache()
EXPERIMENT = 'caterpie'
experiment = EXPERIMENT
device='cuda:6'
# Name this model training run for wandb
# wandb_project_name = 'j_{}'.format(datetime.date.today().strftime('%Y%m%d'))
wandb_project_name = 'caterpie'

id='static_fine'

C = True
CM = False

sigma = 0.1
n_layers=5
n_channels=15
kernel_size=15

wandb_run_name = '30-{}-{}-{}-{}-{}'.format(id, n_layers, n_channels, kernel_size, sigma)
# Timing information for single forward pass
samplerate = 30
num_before = 22
num_after = 7

# Model parameters random seed
seed= 1992
torch.random.manual_seed(seed)

h5_filepath = '/home/jbmelander/caterpie_30hz_finegrain.h5'
direction = 'static'
cluster_file = '/home/jbmelander/caterpie_clusters.txt'
clusters = asmv.get_logged_clusters(cluster_file)


# Set some meta-training parameters

epoch_save_interval = 3

# Model training parameters
batch_size= 128 
learning_rate = 1e-6
weight_decay = 1e-6
num_epochs = 100

# Select training and validation data
train_series = asmv.get_series([5,6])
test_series = asmv.get_series([5, 6], 1)
train_idxs = [30*samplerate, -1]
test_idxs = [0, 30*samplerate]

train_dataset = asmv.CorticalDataset(h5_filepath, 
                                    train_series,
                                    clusters=clusters,
                                    num_before=num_before,
                                    num_after=num_after,
                                    stimulus_key=direction,
                                    start_idx=train_idxs[0],
                                    end_idx=train_idxs[1],
    color=C,
    color_mean=CM)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_dataset = asmv.CorticalDataset(h5_filepath, 
                                   test_series,
                                   clusters=clusters,
                                   num_before=num_before,
                                   num_after=num_after,
                                   stimulus_key=direction,
                                   start_idx=test_idxs[0],
                                   end_idx=test_idxs[1],
    color=C,
    color_mean=CM)
test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Initialize run and make directories
wandb.init(project=wandb_project_name, name=wandb_run_name)
save_dir = f'/home/jbmelander/Models/{wandb_project_name}/{wandb.run.name}'
if os.path.exists(save_dir):
    input('Folder already exists, press enter to overwrite or ctrl-c to cancel')
os.makedirs(save_dir, exist_ok=True)
checkpoint_dir = os.path.join(save_dir, 'checkpoints')
os.makedirs(checkpoint_dir, exist_ok=True)

# Initialize modLCNel parameters
if C:
    if CM:
        model = asmv.CNN2(dataset=train_dataset, num_layers=n_layers, num_channels=n_channels, kernel_sizes=kernel_size, padding=6).to(device)
    else:
        print('Using Color!!!!!!!!')
        print('Using Color!!!!!!!!')
        print('Using Color!!!!!!!!')
        print('Using Color!!!!!!!!')
        print('Using Color!!!!!!!!')
        print('Using Color!!!!!!!!')
        print('Using Color!!!!!!!!')
        model = asmv.CNN3(dataset=train_dataset, num_layers=n_layers, num_channels=n_channels, kernel_sizes=kernel_size, padding=6, gaussian_sigma=sigma).to(device)


# Initialize loss function and optimizer
loss_fn = nn.PoissonNLLLoss(log_input=False, full=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

# Save initial model
torch.save(model, os.path.join(save_dir, 'model_init.pt'))

best_max_corr = 0
best_median_corr = 0
best_mean_corr = 0

for epoch in range(num_epochs):
   model.eval()
   r, p, corrs = asmv.gather_responses(model, test_dataset, device=device)
    
   for cluster in r.keys():
       plt.plot(r[cluster], 'k')
       plt.plot(p[cluster], 'r')
       plt.title(f'Corr: {corrs[cluster]:.2f}')
       plt.savefig('/home/jbmelander/Models/{}/{}/{}.png'.format(wandb_project_name, wandb.run.name, cluster))
       plt.close()

   _corrs = [] 
   for cell, corr in corrs.items():
       print('Cluster: {} Corr: {}'.format(cell, corr))
       _corrs.append(corr)
   corrs = np.array(_corrs)

   # Set nans to 0
   corrs[np.isnan(corrs)] = 0


#    # log epoch metrics
   lr = optimizer.param_groups[0]['lr']
   wandb.log({
       'learning_rate': lr,
       'max_corr': np.max(corrs),
       'min_corr': np.min(corrs),
       'mean_corr': np.mean(corrs),
       'epoch': epoch,
       'median_corr': np.median(corrs)
       })

   if np.max(corrs) > best_max_corr:
       best_max_corr = np.max(corrs)
       torch.save(model, os.path.join(save_dir, 'best_max_corr.pt'))
   if np.nanmedian(corrs) > best_median_corr:
       best_median_corr = np.nanmedian(corrs)
       torch.save(model, os.path.join(save_dir, 'best_median_corr.pt'))
   if np.nanmean(corrs) > best_mean_corr:
       best_mean_corr = np.nanmean(corrs)
       torch.save(model, os.path.join(save_dir, 'best_mean_corr.pt'))
    
   clusters = train_dataset.clusters

#    #train 
   model.train()
   current_batch = 0 # for monitoring progress

   for i, data in tqdm.tqdm(enumerate(train_loader)):
       optimizer.zero_grad()

       stimulus, response = data[0].to(device).float(), data[1].to(device)
       output = model(stimulus)
        
       # model performance loss
       poisson_loss = loss_fn(output, response)
       l1_output_loss = output.abs().sum() * 1e-7

       for name, param in model.named_parameters():
           if 'weight' in name:
               l1_loss = param.abs().sum() * 1e-7
        

       total_loss = poisson_loss # + l1_output_loss + l1_loss
       # update model
       total_loss.backward()

       if i % 10 == 0:
           print(total_loss.item())
       optimizer.step()
        
       # log batch metrics
       wandb.log({'poisson_loss': poisson_loss.item(), 
                  'l1_output_loss': l1_output_loss.item(),
                  'l1_loss': l1_loss.item(),
                  'total_loss': total_loss.item(),
                  'epoch': epoch})
        
       # print progress
   # save model
   if epoch % epoch_save_interval == 0:
       torch.save(model, os.path.join(checkpoint_dir, f'epoch_{epoch}.pt'))

   # # # update learning rate according to schedule
   scheduler.step(total_loss.item())
