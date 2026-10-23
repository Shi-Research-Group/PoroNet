# PoroNet: H2 adsorption on pore graphs


### This Jupyter notebook demonstrates PoroNet, a graph neural network built on pore graphs generated using Mofography. The example model predicts H₂ adsorption at 160 K and 5 bar. PoroNet predicts gas adsorption and reveals the contributions of individual pores, providing interpretable insights into adsorption behavior.

For more details, see
https://pubs.acs.org/jctcce/article/22/11/5716/5185417/PoroNet-An-Intrinsically-Interpretable-Pore-Graph

Requirements

- python==3.11.9
- torch==2.2.1
- dgl==2.4.0
- numpy==1.23.5
- pandas==2.2.2
- scipy==1.13.1
- networkx==3.2.1
- scikit-learn==1.5.1
- scikit-image==0.24.0
- optuna==3.6.1
- openpyxl==3.1.5
- plotly==5.19.0

Import Packages

``` python
import numpy as np 
import os 
import sys
import pandas as pd
import pickle
from skimage.measure import regionprops
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv
from torch.optim import Adam
import optuna
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import networkx as nx
from collections import defaultdict
from scipy.stats import gaussian_kde

torch.set_num_threads(8)
```

Set Random Seeds

``` python
torch.manual_seed(1)
np.random.seed(1) 
```

Import Pore Graphs

``` python
import os, pickle, urllib.request
PKL = "Pore_Graph_for_H2.pkl"
URL = "https://github.com/mofography/mofography/releases/download/poronet-data-v1/Pore_Graph_for_H2.pkl"
if not os.path.exists(PKL):
    urllib.request.urlretrieve(URL, PKL)
RAG_2D = pickle.load(open(PKL, "rb"))
```

``` python
#Divide the pore graphs into training, validation, and testing dataset
RAG_Training=RAG_2D[0:900]
RAG_Validation=RAG_2D[900:990]
RAG_Testing=RAG_2D[990:1990] 

print([len(RAG_Training),len(RAG_Validation),len(RAG_Testing)])
```

    [900, 90, 1000]

``` python
#An example of node feature
RAG_Training[0].nodes[1]
```

    {'frac_pos': array([0.97, 0.  , 0.03]),
     'pocket_radius': 12.69157886505127,
     'cell_volume': 1.3210617229202183e-23,
     'pixel_counts': 111515,
     'total_pixel_amounts': 128588,
     'pixel_ratio': 0.8672271129498864,
     'eh_y': array([0.        , 0.        , 0.        , 0.        , 0.        ,
            0.0008519 , 0.01087746, 0.04914137, 0.08951262, 0.11713222,
            0.54011568, 0.19236874]),
     'eh_x': array([-inf, -9.5, -8.5, -7.5, -6.5, -5.5, -4.5, -3.5, -2.5, -1.5, -0.5,
             inf])}

Process Graphs and Extract Features

``` python
# Convert graphs to DGL graphs and set node features
def Process_Graphs_Labels(RAGs):
  DGL_Graphs = []
  for g in RAGs:
    if g is not None:
        # dgl.from_networkx relabels nodes in sorted order, so read node data in sorted order too
        nodes = sorted(g.nodes)
        DGL_g = dgl.from_networkx(g)
        DGL_g = dgl.add_self_loop(DGL_g)  # Add self-loops
        DGL_g.ndata['feat'] = torch.tensor(np.array([g.nodes[n]['eh_y'] for n in nodes]))
        DGL_g.ndata['maxima_radii'] = torch.tensor([g.nodes[n]['pocket_radius'] for n in nodes])
        DGL_g.ndata['cell_volume'] = torch.tensor([g.nodes[n]['cell_volume'] for n in nodes])
        DGL_g.ndata['pixel_ratio'] = torch.tensor([g.nodes[n]['pixel_ratio'] for n in nodes])
        DGL_Graphs.append(DGL_g)
  return DGL_Graphs

# Get DGL graphs and Labels
DGL_Graphs_Training=Process_Graphs_Labels(RAG_Training)
DGL_Graphs_Validation=Process_Graphs_Labels(RAG_Validation)
DGL_Graphs_Testing=Process_Graphs_Labels(RAG_Testing)

# Convert list of DGL graphs into a single batched graph
Batched_Graph_Training = dgl.batch(DGL_Graphs_Training)
Batched_Graph_Validation=dgl.batch(DGL_Graphs_Validation)
Batched_Graph_Testing = dgl.batch(DGL_Graphs_Testing)

# Extract node features for all graphs
Features_Training = Batched_Graph_Training.ndata['feat'].float()
Features_Validation = Batched_Graph_Validation.ndata['feat'].float()
Features_Testing = Batched_Graph_Testing.ndata['feat'].float()
```

Grab Cell Volumes, Pore Volume, and Pore Pixel Ratio

``` python
def Grab_Cell_Volume(RAGs):
    Cell_Volume=[]
    for i,g in enumerate(RAGs):
        vol=g.nodes[1]['cell_volume']
        Cell_Volume.append(vol)
    return torch.tensor(Cell_Volume).unsqueeze(1)

Cell_Volume_Training=Grab_Cell_Volume(RAG_Training)
Cell_Volume_Validation=Grab_Cell_Volume(RAG_Validation) 
Cell_Volume_Testing=Grab_Cell_Volume(RAG_Testing)
```

``` python
def Grab_Pore_Volume(RAGs):
    Pore_Volume=[]
    for i,g in enumerate(RAGs):
        for x in sorted(g.nodes):  # same node order as DGL and Extract_Pixel_Ratio
            cell_vol=g.nodes[x]['cell_volume']
            pore_ratio=g.nodes[x]['pixel_ratio']
            pore_vol=cell_vol*pore_ratio
            Pore_Volume.append(pore_vol)
    return torch.tensor(Pore_Volume).unsqueeze(1)

Pore_Volume_Training=Grab_Pore_Volume(RAG_Training)
Pore_Volume_Validation=Grab_Pore_Volume(RAG_Validation)
Pore_Volume_Testing=Grab_Pore_Volume(RAG_Testing)
```

``` python
def Extract_Pixel_Ratio(RAGs):
 pixel_ratio_list=[]
 for i, G in enumerate(RAGs):
    for n in sorted(G.nodes):
        pixel_ratio_list.append(G.nodes[n]['pixel_ratio'])
 return pixel_ratio_list 

Pixel_Ratio_List_Training=torch.tensor(Extract_Pixel_Ratio(RAG_Training)).float().view(-1,1)
Pixel_Ratio_List_Validation=torch.tensor(Extract_Pixel_Ratio(RAG_Validation)).float().view(-1,1)
Pixel_Ratio_List_Testing=torch.tensor(Extract_Pixel_Ratio(RAG_Testing)).float().view(-1,1)
```

Import MOF-level Labels

``` python
Data_all=pd.read_excel('160K_5bar-mof-gL.xlsx').to_numpy()[:,1].reshape(-1,1)

Labels_Training_mof=Data_all[0:900]
Labels_Validation_mof=Data_all[900:990]
Labels_Testing_mof=Data_all[990:1990]

Labels_Training_mof = np.array([[float(x[0])] for x in Labels_Training_mof])
Labels_Validation_mof = np.array([[float(x[0])] for x in Labels_Validation_mof])
Labels_Testing_mof = np.array([[float(x[0])] for x in Labels_Testing_mof])

Labels_Training_mof=torch.tensor(Labels_Training_mof)
Labels_Validation_mof=torch.tensor(Labels_Validation_mof)
Labels_Testing_mof=torch.tensor(Labels_Testing_mof)
```

Define PoroNet

``` python
class PoroNet(nn.Module):
    def __init__(self,input_size,hidden_sizes,hidden_activations,dropout):
        super(PoroNet,self).__init__()
        layers=[]
        in_feat=input_size
         
        for i,hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(in_feat,hidden_size))
            hidden_activation=hidden_activations[i]
            if hidden_activation=='relu':
                layers.append(nn.ReLU())
            in_feat=hidden_size

        layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(in_feat,1))
        self.model = nn.Sequential(*layers)
         
         
    def forward(self,g,input_feature):

        x=self.model(input_feature)

        return x
```

Determine the Hyperparameters

``` python
#5 fold
cv=5

#maximum of training epoch
num_epoch=5000
```

``` python
def objective(trial):
    # Suggest values for the hyperparameters
    num_patience=trial.suggest_int('num_patience', 10, 2000)
    num_layers=trial.suggest_int('num_layers', 1, 3)
    hidden_sizes = [trial.suggest_int(f'hidden_size{i}', 16, 128) for i in range(num_layers)]
    hidden_activations=[trial.suggest_categorical(f'hidden_activation{i}',['relu','none'])for i in range(num_layers)]
    dropout = trial.suggest_uniform('dropout', 0.2, 0.4)
    initial_lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-9, 1e-3)
    factor=trial.suggest_uniform('factor',0.1, 0.9)
    

    # K-Fold CV
    kf = KFold(n_splits=cv, shuffle=False)
    mae_list = []
    for fold, (train_index, test_index) in enumerate(kf.split(range(len(RAG_Training)))):
     
     # Extract pore graphs and labels according to index
     RAG_training = [RAG_Training[i] for i in train_index]
     Labels_training = [Labels_Training_mof[i] for i in train_index]
     RAG_validation = [RAG_Training[i] for i in test_index]
     Labels_validation = [Labels_Training_mof[i] for i in test_index]
     
     Labels_training = np.array([float(x[0]) for x in Labels_training])
     Labels_validation = np.array([float(x[0]) for x in Labels_validation])
    
     # Get DGL graphs and Labels
     DGL_Graphs_training=Process_Graphs_Labels(RAG_training)
     DGL_Graphs_validation=Process_Graphs_Labels(RAG_validation)
     # Convert labels to tensor
     Labels_training=torch.tensor(Labels_training).view(-1,1)
     Labels_validation=torch.tensor(Labels_validation).view(-1,1)
     
     # Convert list of DGL graphs into a single batched graph
     Batched_Graph_training = dgl.batch(DGL_Graphs_training)
     Batched_Graph_validation=dgl.batch(DGL_Graphs_validation)
     # Extract node features for all graphs
     Features_training = Batched_Graph_training.ndata['feat'].float()
     Features_validation = Batched_Graph_validation.ndata['feat'].float()
        
     Pixel_Ratio_List_training=torch.tensor(Extract_Pixel_Ratio(RAG_training)).float().view(-1,1)
     Pixel_Ratio_List_validation=torch.tensor(Extract_Pixel_Ratio(RAG_validation)).float().view(-1,1)
        
     # Initialize the model and optimizer
     model = PoroNet(Features_training.size(1),hidden_sizes,hidden_activations,dropout)
     optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
     scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=factor, patience=0.9* num_patience, verbose=True)
        
     # Initialize early stopping parameters
     best_val_loss = float('inf') 
     stop_patience = num_patience 
     counter = 0   
     early_stopped = False  
     best_epoch_loss = None     
     
     # Training loop
     for epoch in range(num_epoch):
         model.train()
         Embeddings_training_1 = model(Batched_Graph_training,Features_training)  # Forward propagation, get embeddings
         Embeddings_training_2 = Embeddings_training_1 * Pixel_Ratio_List_training  #Times the density of pores
         Batched_Graph_training.ndata['h'] = Embeddings_training_2  # Assign embeddings to 'h' field
         Logits_training = dgl.sum_nodes(Batched_Graph_training, 'h')  # Sum the embeddings across nodes
         Loss_training = F.l1_loss(Logits_training, Labels_training)  # Compute loss
         optimizer.zero_grad()
         Loss_training.backward()
         optimizer.step()
        
    
         model.eval()  # Set model to evaluation mode
         with torch.no_grad():
          Embeddings_validation_1 = model(Batched_Graph_validation,Features_validation) 
          Embeddings_validation_2 = Embeddings_validation_1 * Pixel_Ratio_List_validation  
          Batched_Graph_validation.ndata['h'] = Embeddings_validation_2  
          Logits_validation = dgl.sum_nodes(Batched_Graph_validation, 'h')  
          Loss_validation = F.l1_loss(Logits_validation, Labels_validation)  
          scheduler.step(Loss_validation)
         
             
         # Early stopping
         if Loss_validation < best_val_loss:
          best_val_loss = Loss_validation
          best_epoch_loss = Loss_validation.item()
          counter = 0
         else:
            counter += 1
            
         if counter >= stop_patience:
            print(f'Early stopping at epoch {epoch+1}')
            early_stopped = True
            mae_list.append(best_epoch_loss)
            break
        
     if not early_stopped:  # If not early stopped, record the loss of the last epoch
            mae_list.append(Loss_validation.item())

    mae_average = np.mean(mae_list) 
    
    if trial.should_prune():
     raise optuna.exceptions.TrialPruned()
    mae_average=np.mean(mae_list)
    trial.set_user_attr('mae_list', mae_list)

    return mae_average
```

``` python
sampler = optuna.samplers.TPESampler(seed=1)
study = optuna.create_study(sampler=sampler,direction='minimize', pruner=optuna.pruners.HyperbandPruner())
study.optimize(objective, n_trials=100)

print('Best trial:')
trial = study.best_trial
print('  Trial Number: ', trial.number)
print('  Value: ', trial.value)
print('  Params: ')
for key, value in trial.params.items():
    print('    {}: {}'.format(key, value)) 
print('  Fold MAEs: ', trial.user_attrs['mae_list'])
```

Train the Model

``` python
num_patience=486
model = PoroNet(Features_Testing.size(1),hidden_sizes=[114],hidden_activations=['relu'],dropout=0.2132217405513009)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0019286667936327432, weight_decay=1.6685073137586203e-09)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor= 0.10056880106742247, patience=0.9*num_patience, verbose=True)
```

``` python
# Initialize early stopping parameters
best_val_loss = float('inf') 
stop_patience =num_patience # Number of epochs to wait for improvement
counter = 0  # Counter for patience  
# Initialize the loss of training and validation set
Loss_Training_List=[]
Loss_Validation_List=[]

# Training loop
for epoch in range(num_epoch):
        model.train()
        Embeddings_Training_1 = model(Batched_Graph_Training,Features_Training) 
        Embeddings_Training_2 = Embeddings_Training_1 * Pixel_Ratio_List_Training  
        Batched_Graph_Training.ndata['h'] = Embeddings_Training_2  
        Logits_Training = dgl.sum_nodes(Batched_Graph_Training, 'h') 
        Loss_Training = F.l1_loss(Logits_Training, Labels_Training_mof) 
        optimizer.zero_grad()
        Loss_Training_List.append(Loss_Training.item())
        Loss_Training.backward()
        optimizer.step()
        
        # Set model to evaluation mode
        model.eval() 
        with torch.no_grad():
         Embeddings_Validation_1 = model(Batched_Graph_Validation,Features_Validation) 
         Embeddings_Validation_2 = Embeddings_Validation_1 * Pixel_Ratio_List_Validation 
         Batched_Graph_Validation.ndata['h'] = Embeddings_Validation_2  
         Logits_Validation = dgl.sum_nodes(Batched_Graph_Validation, 'h')  
         Loss_Validation = F.l1_loss(Logits_Validation, Labels_Validation_mof)  
         Loss_Validation_List.append(Loss_Validation.item())
         scheduler.step(Loss_Validation)
        # Early stopping
        if Loss_Validation < best_val_loss:
         best_val_loss = Loss_Validation
         counter = 0
        else:
         counter += 1
        if counter >= stop_patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
```

    Early stopping at epoch 3411

Test the model at the MOF level

``` python
model.eval()  
with torch.no_grad():
    Predictions_Testing_pore = model(Batched_Graph_Testing,Features_Testing)  
    Predictions_Testing_pore_2 = Predictions_Testing_pore * Pixel_Ratio_List_Testing
    Batched_Graph_Testing.ndata['h'] = Predictions_Testing_pore_2  
    Predictions_Testing_mof = dgl.sum_nodes(Batched_Graph_Testing, 'h')
```

``` python
#MOF-level Predictions (g/L), first 10 testing MOFs
Predictions_Testing_mof[:10]
```

    tensor([[1.4284],
            [0.9900],
            [1.2906],
            [1.2496],
            [0.9949],
            [4.5672],
            [1.9631],
            [2.1292],
            [1.0040],
            [1.3258]])

``` python
#Calculate R² and MAE at the MOF level
r2_mof = r2_score(Labels_Testing_mof, Predictions_Testing_mof)
Loss_Testing_mof = F.l1_loss(Predictions_Testing_mof, Labels_Testing_mof)
r2_mof, Loss_Testing_mof
```

    (0.98401606474642, tensor(0.0431, dtype=torch.float64))

``` python
#Parity plot
import plotly.graph_objects as go
import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import gaussian_kde

# Calculate R²
r2 = r2_score(Labels_Testing_mof, Predictions_Testing_mof)
# Calculate point density
x = Labels_Testing_mof.flatten()
y = Predictions_Testing_mof.flatten()
xy = np.vstack([x, y])
density = gaussian_kde(xy)(xy)  # Density estimate at each point

# Create scatter plot colored by density
scatter = go.Scatter(
    x=x,
    y=y,
    mode='markers',

    marker=dict(
    size=8,
    color=density,  # Use density as the color value
    colorscale=[
    [0.0, '#F6F2FF'],   # 
    [0.1, '#a066c2'],   # soft purple
    [0.2, '#6b8ed6'],   # soft blue
    [0.4, '#5fbba1'],   # soft green
    [0.6, '#e1d05f'],   # soft yellow
    [0.8, '#f7a072'],   # soft orange
    [0.9, '#e35c5c'],   # soft red
    [1.0, '#640206']    # dark red
    ],
    opacity=1,
    
    colorbar=dict(
    title='Point Density',
    title_font=dict(color='black',size=35, family='Arial'),
    title_side='right',

    tickvals=[],

    outlinewidth=0
    ),
    line=dict(width=0),
   
    ),
    text=[f'Actual: {a}, Predicted: {p}' for a, p in zip(x, y)],
    hoverinfo='text'
)

# Create a line for perfect parity
parity_line = go.Scatter(
    x=[-1,60],
    y=[-1,60],
    mode='lines',
    line=dict(color='black', dash='dash', width=4)  # Dashed line style and color
)

# Create a layout
layout = go.Layout(

    xaxis=dict(
    title='GCMC Result (g/L)',
    range=[0, 10],
    tickfont=dict(color='black', size=40),
    titlefont=dict(color='black', size=40),
    linecolor='black',
    showline=True,
    linewidth=4,
    dtick=2,
    ticks='outside',
    tickwidth=4,
    tickcolor='black',
    ticklen=8,
    mirror=True,             # Mirror the axis line on the opposite side
    showticklabels=True,     # Show tick labels
    ticksuffix=' ',          # Add spacing after tick labels
    ),

    yaxis=dict(
    title='PoroNet-Base Result (g/L)',
    range=[0, 10],
    tickfont=dict(color='black', size=40),
    titlefont=dict(color='black', size=40),
    linecolor='black',
    showline=True,
    linewidth=4,
    dtick=2,
    ticks='outside',
    tickwidth=4,
    tickcolor='black',
    ticklen=8,
    mirror=True,             # Mirror the axis line on the opposite side
    showticklabels=True,     # Show tick labels
    ticksuffix=' ',          # Add spacing after tick labels
    ),
    showlegend=False,  # Hide legend
    height=800,
    width=830,
    font=dict(
        family='Arial',  # Global font: Arial
        size=35  # Global font size
    ),
    annotations=[
        dict(
            x=0.01,  # Text x position (0-1)
            y=0.98,  # Text y position (0-1)
            xref='paper',  # x reference frame
            yref='paper',  # y reference frame
            text=f'H<sub>2</sub> Uptake (160K 5bar)<br>1000 Testing MOFs',  # Text content
            showarrow=False,  # No arrow
            font=dict(size=35, family='Arial', color='black'),  # Font settings
            align='center'  # Center-align text
        ),
        dict(
            x=0.98,  # Text x position (0-1)
            y=0,  # Text y position (0-1)
            xref='paper',  # x reference frame
            yref='paper',  # y reference frame
            text=f'R² = {r2:.2f}<br>MAE = {Loss_Testing_mof.item():.2f} g/L',  # Text content
            showarrow=False,  # No arrow
            font=dict(size=35, family='Arial', color='black'),  # Font settings
            align='center'  # Center-align text
        )        
        
    ],
    plot_bgcolor='white',  # White plot-area background
    paper_bgcolor='white'  # White canvas background
)

# Create a Figure and add the scatter plot and parity line
fig = go.Figure(data=[scatter, parity_line], layout=layout)

# Static PNG so the plot also shows on GitHub and in the docs
fig.show(renderer="png", width=830, height=800)
```

![](PoroNet-Base_160K_5bar_files/figure-commonmark/cell-21-output-1.png)

``` python
#Pore-level adsorption capacity can appear as an emergent property (g/L),first 10 testing pores
Predictions_Testing_pore[:10]
```

    tensor([[1.6352],
            [1.0655],
            [1.0409],
            [1.0495],
            [1.0251],
            [1.0408],
            [1.0144],
            [1.0313],
            [1.0321],
            [1.0259]])
