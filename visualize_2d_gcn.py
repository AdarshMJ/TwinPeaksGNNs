import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit, LargestConnectedComponents
from model import SimpleGCN  # Import the GCN model
import random
import os
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import torch.nn.functional as F
import networkx as nx
from torch_geometric.nn import GATv2Conv  # Add this import at the top with other imports


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True

def train_2d_model(dataset_name='Cora', model_type='gcn', hidden_dim=2, num_epochs=1000, lr=0.001, seed=42):
    """Train a GNN model (GCN or GATv2) with 2D hidden representations"""
    set_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    data = dataset[0]
    # data = np.load('data/texas.npz')
    # print("Converting to PyG dataset...")
    # x = torch.tensor(data['node_features'], dtype=torch.float)
    # y = torch.tensor(data['node_labels'], dtype=torch.long)
    # edge_index = torch.tensor(data['edges'], dtype=torch.long).t().contiguous()
    # train_mask = torch.tensor(data['train_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    # val_mask = torch.tensor(data['val_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    # test_mask = torch.tensor(data['test_masks'], dtype=torch.bool).transpose(0, 1).contiguous()
    # num_classes = len(torch.unique(y))
    # data = Data(x=x, edge_index=edge_index, y=y, 
    #             train_mask=train_mask[0],
    #             val_mask=val_mask[0],
    #             test_mask=test_mask[0],
    #             num_classes=num_classes)
    
    print(f"Selecting the LargestConnectedComponent..")
    transform = LargestConnectedComponents()
    data = transform(data)
    transform2 = RandomNodeSplit(split="test_rest",num_splits=1,num_test=0.3,num_val=0.1)
    data = transform2(data)
    print(data)
    data = data.to(device)
    
    # Initialize model based on type
    if model_type.lower() == 'gcn':
        model = SimpleGCN(
            num_features=data.num_features,
            num_classes=data.num_classes,
            hidden_channels=hidden_dim,
            num_layers=2
        ).to(device)
    elif model_type.lower() == 'gatv2':
        model = SimpleGATv2(
            num_features=dataset.num_features,
            num_classes=dataset.num_classes,
            hidden_channels=hidden_dim,
            num_layers=2,
            heads=1  # Using 1 attention head to keep 2D representations
        ).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            model.eval()
            pred = out.argmax(dim=1)
            train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).sum().item() / data.train_mask.sum().item()
            print(f'Epoch {epoch+1:3d}, Loss: {loss.item():.4f}, Train Acc: {train_acc:.4f}')
            model.train()
    
    return model, data

def clean_dataset_name(dataset_name):
    """Convert dataset filename to clean display name"""
    # Remove file extension
    name = dataset_name.replace('.npz', '')
    # Capitalize first letter and replace underscores with spaces
    name = name.replace('_', ' ').title()
    return name

def visualize_hidden_representations(model, data, save_dir, dataset_name, num_epochs):
    """Visualize 2D hidden representations"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    clean_name = clean_dataset_name(dataset_name)
    
    model.eval()
    x = data.x
    
    with torch.no_grad():
        representations = []
        weight_matrices = []
        
        # Move x to the same device as the model
        x = data.x
        
        # Initial features
        representations.append(x)
        
        # First layer weights and output
        if hasattr(model.convs[0], 'lin'):
            W1 = model.convs[0].lin.weight.cpu().detach().numpy()
        else:
            W1 = model.convs[0].lin_r.weight.cpu().detach().numpy()
        weight_matrices.append(W1.T)
        
        # First layer output
        x = model.convs[0](x, data.edge_index)
        x = F.relu(x)
        representations.append(x)
        
        # Second layer weights and output
        if hasattr(model.convs[1], 'lin'):
            W2 = model.convs[1].lin.weight.cpu().detach().numpy()
        else:
            W2 = model.convs[1].lin_r.weight.cpu().detach().numpy()
        weight_matrices.append(W2.T)
        
        # Second layer output
        x = model.convs[1](x, data.edge_index)
        representations.append(x)
        
        # Convert to numpy - ensure CPU conversion first
        representations = [r.cpu().detach().numpy() for r in representations]
        
        # Create visualization with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Plot weight matrices (Features - columns of W)
        for i, W in enumerate(weight_matrices):
            ax = axes[0, i]
            ax.scatter(W[:, 0], W[:, 1], c='blue', alpha=0.6, s=20)
            ax.set_title(f'{clean_name} - Layer {i+1}\nFeatures (columns of W)')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        # Plot hidden vectors (only layers 1 and 2) in bottom row
        for i in range(2):
            ax = axes[1, i]
            hidden = representations[i + 1]
            # Move train_mask to CPU if needed
            train_mask = data.train_mask.cpu().numpy()
            hidden_train = hidden[train_mask]
            
            for h in hidden_train:
                ax.plot([0, h[0]], [0, h[1]], 'r-', alpha=0.3, linewidth=0.5)
            ax.scatter(hidden_train[:, 0], hidden_train[:, 1], c='red', alpha=0.6, s=20)
            
            ax.set_title(f'{clean_name} - Layer {i + 1}\nHidden Vectors')
            ax.set_xlabel('Dimension 1')
            ax.set_ylabel('Dimension 2')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.suptitle(f'{clean_name} - Hidden Representations Across Layers\nTop: Features (W), Bottom: Hidden Vectors', 
                    y=1.02, fontsize=16)
        plt.tight_layout()
        
        # Save visualization
        plt.savefig(save_dir / f'{clean_name}_epochs{num_epochs}_2d_hidden_representations_{timestamp}.png',
                   bbox_inches='tight', dpi=300)
        plt.close()

# Add new GATv2 model class
class SimpleGATv2(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, num_layers, heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GATv2Conv(num_features, hidden_channels, heads=heads))
        
        # Output layer
        self.convs.append(GATv2Conv(hidden_channels * heads, num_classes, heads=1))

    def forward(self, x, edge_index):
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x

if __name__ == "__main__":
    # Update parameters
    DATASET = 'Cora'
    MODEL_TYPE = 'gatv2'  # or 'gcn'
    HIDDEN_DIM = 2
    NUM_EPOCHS = 3000
    LEARNING_RATE = 0.01
    SEED = 42
    SAVE_DIR = Path('visualization_results')
    
    print(f"Training 2D {MODEL_TYPE.upper()} on {DATASET}...")
    model, data = train_2d_model(
        dataset_name=DATASET,
        model_type=MODEL_TYPE,
        hidden_dim=HIDDEN_DIM,
        num_epochs=NUM_EPOCHS,
        lr=LEARNING_RATE,
        seed=SEED
    )
    
    print("Generating visualizations...")
    visualize_hidden_representations(model, data, SAVE_DIR, DATASET, NUM_EPOCHS)
    print(f"Visualizations saved in {SAVE_DIR}") 
