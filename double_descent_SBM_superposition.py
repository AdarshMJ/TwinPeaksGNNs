#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time
from torch_geometric.data import Data
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv, PairNorm
import random
import os
import logging
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from matplotlib.gridspec import GridSpec
import argparse
import pandas as pd
from torch_geometric.explain import Explainer, GNNExplainer
import matplotlib
import networkx as nx
from sklearn.manifold import TSNE
import matplotlib.gridspec as gridspec
matplotlib.use('Agg')  # Use the Agg backend (non-interactive)

# Set up logging
def setup_logging(save_dir, config):
    """Set up logging with comprehensive parameter tracking and suppress matplotlib info messages"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = save_dir / f'superposition_{timestamp}.log'
    
    # Create a custom formatter
    formatter = logging.Formatter('%(message)s')  # Simplified formatter for cleaner logs
    
    # Set up handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure root logger for our application logs
    logging.root.setLevel(logging.INFO) # Keep our application logs at INFO level
    logging.root.handlers = [file_handler, console_handler]
    
    # Suppress matplotlib INFO messages
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING) # Set matplotlib logger to WARNING or higher
    
    # Log header
    logging.info("="*80)
    logging.info(f"Superposition Analysis - {timestamp}")
    logging.info("="*80 + "\n")
    
    # Log configuration parameters in groups
    logging.info("MODEL CONFIGURATION")
    logging.info("-"*50)
    logging.info(f"Model Type        : {config['model_type']}")
    logging.info(f"Hidden Dimension  : {config['hidden_dim']}")
    logging.info(f"Number Features   : {config['num_features']}")
    logging.info(f"Number Classes    : {config['num_classes']}")
    logging.info(f"Dropout Rate      : {config['dropout_rate']}")
    if config['model_type'] == "gat":
        logging.info(f"GAT Heads        : {config['gat_heads']}")
    logging.info("")
    
    logging.info("DATASET CONFIGURATION")
    logging.info("-"*50)
    logging.info(f"Feature Sparsity  : {config['sparsity']}")
    logging.info(f"Number of Blocks  : {config['num_blocks']}")
    logging.info(f"Dataset Sizes     : {config['sizes']}")
    logging.info("")
    
    logging.info("TRAINING CONFIGURATION")
    logging.info("-"*50)
    logging.info(f"Number of Epochs  : {config['num_epochs']}")
    logging.info(f"Learning Rate     : {config['base_lr']}")
    logging.info(f"Random Seed       : {config['seed']}")
    logging.info("")
    
    logging.info("VISUALIZATION CONFIGURATION")
    logging.info("-"*50)
    logging.info(f"Small Threshold   : {config['small_threshold']}")
    logging.info(f"Large Threshold   : {config['large_threshold']}")
    logging.info(f"Selected Sizes    : {config['selected_sizes']}")
    logging.info("\n" + "="*80 + "\n")

# Model definition
class SimpleGCN(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        #self.pairnorm = PairNorm(scale=1.0)
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        #x = self.pairnorm(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

class SimpleGAT(nn.Module):
    def __init__(self, num_features, num_classes, hidden_channels, heads):
        super().__init__()
        self.conv1 = GATv2Conv(num_features, hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(hidden_channels, num_classes, heads=heads)
        #self.pairnorm = PairNorm(scale=1.0)
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(0.0)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        #x = self.pairnorm(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x

def generate_sbm_dataset(num_nodes, num_features=100, num_classes=2, num_blocks=5, sparsity=0.95, seed=42):
    """Generate a Stochastic Block Model dataset with sparse features"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Generate block assignments
    nodes_per_block = num_nodes // num_blocks
    remaining_nodes = num_nodes % num_blocks
    block_sizes = [nodes_per_block + (1 if i < remaining_nodes else 0) for i in range(num_blocks)]
    block_assignments = []
    for block_idx, size in enumerate(block_sizes):
        block_assignments.extend([block_idx] * size)
    
    # Map 5 blocks to 2 classes (this creates interesting structure)
    y = torch.tensor(block_assignments) % num_classes
    
    # Adjust edge probabilities based on dataset size
    if num_nodes <= 75:
        p_intra = 0.4
        p_inter = 0.05
    elif num_nodes <= 400:
        p_intra = 0.3
        p_inter = 0.02
    else:
        p_intra = 0.2
        p_inter = 0.01
    
    # Generate edges
    edge_index = []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            p = p_intra if block_assignments[i] == block_assignments[j] else p_inter
            if np.random.random() < p:
                edge_index.extend([[i, j], [j, i]])
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    
    # Generate sparse features
    x = torch.zeros(num_nodes, num_features)
    mask = torch.rand(num_nodes, num_features) > sparsity
    values = torch.randn(num_nodes, num_features)
    x[mask] = values[mask]
    
    # Create masks
    indices = np.random.permutation(num_nodes)
    train_size = int(0.6 * num_nodes)
    val_size = int(0.2 * num_nodes)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True
    
    return Data(x=x, edge_index=edge_index, y=y,
                train_mask=train_mask,
                val_mask=val_mask,
                test_mask=test_mask)

def calculate_superposition(hidden_vectors):
    """Calculate superposition matrix (dot products between feature vectors)"""
    norms = torch.norm(hidden_vectors, dim=1, keepdim=True)
    normalized_vectors = hidden_vectors / (norms + 1e-8)
    superposition_matrix = torch.mm(normalized_vectors, normalized_vectors.t())
    return superposition_matrix.cpu().numpy()

def visualize_superposition(models, all_data, sizes, save_dir, device, selected_sizes, model_type):
    """Visualize superposition across dataset sizes"""
    num_selected = len(selected_sizes)
    
    plt.figure(figsize=(15, 5))
    
    sparsity_values = []
    
    # Create a red-blue colormap with white center
    colors = plt.cm.RdBu
    
    for idx, size in enumerate(selected_sizes):
        size_idx = sizes.index(size)
        model = models[size_idx]
        data = all_data[size_idx].to(device)
        
        model.eval()
        with torch.no_grad():
            hidden = model.conv1(data.x, data.edge_index)
            #hidden = model.pairnorm(hidden)
            hidden = F.relu(hidden)
            
            # Calculate superposition
            superposition = calculate_superposition(hidden)
            
            # Calculate sparsity metric (1-S)
            sparsity = 1 - (np.abs(superposition).sum() - superposition.shape[0]) / (superposition.shape[0] * (superposition.shape[0] - 1))
            sparsity_values.append(sparsity)
            
            # Plot superposition matrix
            plt.subplot(1, num_selected, idx + 1)
            im = plt.imshow(superposition, cmap=colors, vmin=-1, vmax=1)
            plt.title(f'n={size}\n1-S={sparsity:.3f}')
            plt.colorbar(im)
            
            # Remove ticks but keep axis labels
            plt.tick_params(axis='both', which='both', length=0)
    
    plt.suptitle('Superposition Analysis Across Dataset Sizes', y=1.05)
    plt.tight_layout()
    
    # Save with higher resolution
    plt.savefig(save_dir / f'superposition_analysis_{model_type}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return sparsity_values

def plot_double_descent_curve(results, save_dir):
    """Plot double descent curve"""
    plt.figure(figsize=(12, 8))
    
    # Plot double descent curve
    dataset_sizes = sorted(results.keys())
    losses = [results[size]['test_loss'] for size in dataset_sizes]
    
    plt.plot(dataset_sizes, losses, 'o-', color='blue', linewidth=2, 
             markersize=8, label='Test Loss')
    
    # Customize plot
    plt.xscale('log')
    plt.xlabel('Dataset size', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add background colors for regimes
    plt.axvspan(min(dataset_sizes), 100, alpha=0.1, color='blue', label='Small Data')
    plt.axvspan(100, 500, alpha=0.1, color='gray', label='Middle Regime')
    plt.axvspan(500, max(dataset_sizes), alpha=0.1, color='green', label='Large Data')
    
    # Add regime labels
    ymax = max(losses) * 1.2
    plt.text(min(dataset_sizes) * 1.2, ymax * 1.05, 'SMALL DATA SETS', 
             ha='left', va='bottom', fontsize=12, fontweight='bold')
    plt.text(100 * 1.2, ymax * 1.05, 'MIDDLE REGIME', 
             ha='left', va='bottom', fontsize=12, fontweight='bold')
    plt.text(500 * 1.2, ymax * 1.05, 'LARGE DATA SETS', 
             ha='left', va='bottom', fontsize=12, fontweight='bold')
    
    plt.ylim(0, ymax)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(save_dir / 'double_descent_curve.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def visualize_polytopes(models, all_data, sizes, save_dir, device, model_type):
    """Visualize features and hidden vectors for selected dataset sizes"""
    # Select specific sizes to visualize
    selected_sizes = [30, 50, 75, 200, 500, 1000]
    num_plots = len(selected_sizes)
    
    # Create figure with 2 rows
    fig, axes = plt.subplots(2, num_plots, figsize=(24, 12))
    
    for plot_idx, size in enumerate(selected_sizes):
        size_idx = sizes.index(size)
        model = models[size_idx]
        data = all_data[size_idx].to(device)
        
        model.eval()
        with torch.no_grad():
            # Get input features
            features = data.x.detach().cpu()
            
            # Add small epsilon to avoid numerical instabilities
            eps = 1e-8
            
            # Standardize and reduce dimensions for features
            scaler = StandardScaler()
            features_std = scaler.fit_transform(features.numpy())
            
            # Add small noise to avoid constant features
            features_std += np.random.normal(0, eps, features_std.shape)
            
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_std)
            features_plot = torch.tensor(features_2d)
            
            # Get hidden representations
            x = model.conv1(data.x, data.edge_index)
            x = F.relu(x)
            hidden = x.detach().cpu()
            
            # Standardize and reduce dimensions for hidden vectors
            hidden_std = scaler.fit_transform(hidden.numpy())
            
            # Add small noise to avoid constant features
            hidden_std += np.random.normal(0, eps, hidden_std.shape)
            
            hidden_2d = pca.fit_transform(hidden_std)
            hidden_plot = torch.tensor(hidden_2d)
            
            # Calculate center (mean) of features
            features_center = features_plot.mean(dim=0)
            
            # Plot features
            axes[0, plot_idx].scatter(features_plot[:, 0], features_plot[:, 1], 
                                    color="blue", alpha=0.6, s=80)  # Increased point size
            
            # Draw lines from center to each point
            for i in range(len(features_plot)):
                axes[0, plot_idx].plot([features_center[0], features_plot[i, 0]], 
                                     [features_center[1], features_plot[i, 1]], 
                                     color="blue", alpha=0.2, linewidth=0.5)
            
            axes[0, plot_idx].set_title(f'n={size}', fontsize=14, pad=10)
            
            # Calculate center (mean) of hidden vectors
            hidden_center = hidden_plot.mean(dim=0)
            
            # Plot hidden vectors
            axes[1, plot_idx].scatter(hidden_plot[:, 0], hidden_plot[:, 1], 
                                    color="red", alpha=0.6, s=80)  # Increased point size
            
            # Draw lines from center to each point
            for i in range(len(hidden_plot)):
                axes[1, plot_idx].plot([hidden_center[0], hidden_plot[i, 0]], 
                                     [hidden_center[1], hidden_plot[i, 1]], 
                                     color="red", alpha=0.2, linewidth=0.5)
            
            # Improve axes appearance
            for ax_row in [axes[0, plot_idx], axes[1, plot_idx]]:
                ax_row.set_xticks([])
                ax_row.set_yticks([])
                ax_row.axvline(0, linestyle="--", color="grey", alpha=0.5, zorder=-1)
                ax_row.axhline(0, linestyle="--", color="grey", alpha=0.5, zorder=-1)
                ax_row.set_aspect('equal')
                
                # Add grid
                ax_row.grid(True, linestyle='--', alpha=0.3)
                
                # Set background color
                ax_row.set_facecolor('white')
                
                # Add box around plot
                for spine in ax_row.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
    
    plt.suptitle('Feature and Hidden Vector Polytopes Across Dataset Sizes', 
                 y=1.02, fontsize=16, fontweight='bold')
    
    # Adjust layout to prevent cutoff
    plt.tight_layout(rect=[0.03, 0, 1, 0.95], h_pad=3, w_pad=2)
    
    # Save with higher resolution
    plt.savefig(save_dir / f'polytope_visualization_{model_type}.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def plot_double_descent_and_polytopes(sizes, losses, models, all_data, save_dir, small_threshold, large_threshold, model_type, device):
    """Create combined plot with double descent curve and polytope visualizations"""
    # Set up the figure with a white background
    fig = plt.figure(figsize=(15, 10), facecolor='white')
    
    # Create grid spec
    gs = plt.GridSpec(2, 6, height_ratios=[2, 1])
    
    # Double descent plot
    ax_dd = fig.add_subplot(gs[0, :])
    ax_dd.set_facecolor('white')
    
    # Customize grid first (so it's behind the plot)
    ax_dd.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax_dd.set_axisbelow(True)  # Put grid behind plot
    
    # Plot double descent with improved styling
    ax_dd.plot(sizes, losses, 'o-', 
              color='#1f77b4',  # Specific blue color
              linewidth=2, 
              markersize=8,
              markeredgewidth=2,
              markeredgecolor='#1f77b4',
              markerfacecolor='#1f77b4',
              label=model_type.upper(),
              zorder=5)  # Ensure line is above grid
    
    # Add background regions with softer colors
    ax_dd.axvspan(min(sizes), small_threshold, alpha=0.1, color='blue', zorder=1)
    ax_dd.axvspan(small_threshold, large_threshold, alpha=0.1, color='gray', zorder=1)
    ax_dd.axvspan(large_threshold, max(sizes), alpha=0.1, color='green', zorder=1)
    
    # Customize spines
    ax_dd.spines['top'].set_visible(False)
    ax_dd.spines['right'].set_visible(False)
    ax_dd.spines['left'].set_linewidth(0.5)
    ax_dd.spines['bottom'].set_linewidth(0.5)
    
    # Add region labels with better positioning
    ymax = max(losses) * 1.2
    ax_dd.text(min(sizes) * 1.2, ymax * 1.05, 'SMALL DATA SETS', 
              ha='left', va='bottom', fontsize=12, fontweight='bold')
    ax_dd.text(small_threshold * 1.2, ymax * 1.05, 'MIDDLE REGIME', 
              ha='left', va='bottom', fontsize=12, fontweight='bold')
    ax_dd.text(large_threshold * 1.2, ymax * 1.05, 'LARGE DATA SETS', 
              ha='left', va='bottom', fontsize=12, fontweight='bold')
    
    # Customize axes
    ax_dd.set_xscale('log')
    ax_dd.set_xlabel('Dataset size', fontsize=12, fontweight='bold')
    ax_dd.set_ylabel('Test Loss', fontsize=12, fontweight='bold')
    
    # Customize legend
    ax_dd.legend(loc='lower right', frameon=True, framealpha=1.0)
    
    # Set y-axis limits with some padding
    ax_dd.set_ylim(0, ymax)
    
    # Selected sizes for polytope visualization
    selected_indices = [
        sizes.index(50),    # small regime
        sizes.index(200),   # middle regime
        sizes.index(1000)   # large regime
    ]
    
    # Create polytope plots (second row)
    for idx, i in enumerate(selected_indices):
        ax_features = fig.add_subplot(gs[1, idx*2])
        ax_hidden = fig.add_subplot(gs[1, idx*2 + 1])
        
        size = sizes[i]
        model = models[i]
        data = all_data[i]
        
        # Get hidden representations
        model.eval()
        with torch.no_grad():
            hidden_vectors = []
            
            # First layer
            x = model.conv1(data.x, data.edge_index)
            x = F.relu(x)
            hidden_vectors.append(x)
            
            # Second layer
            x = model.conv2(x, data.edge_index)
            hidden_vectors.append(x)
            
            # Standardize the data before PCA
            scaler = StandardScaler()
            hidden_std = scaler.fit_transform(hidden_vectors[0].cpu().numpy())
            features_std = scaler.fit_transform(data.x.cpu().numpy())
            
            # Apply PCA
            pca = PCA(n_components=2)
            hidden_2d = pca.fit_transform(hidden_std)
            features_2d = pca.fit_transform(features_std)
        
        # Plot features
        ax_features.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c='blue', alpha=0.5, s=5)  # Increased point size
        ax_features.set_title(f'Features (n={size})')
        
        # Set equal aspect ratio and proper limits
        max_range = max(np.ptp(features_2d[:, 0]), np.ptp(features_2d[:, 1]))
        ax_features.set_aspect('equal')
        ax_features.set_xlim(features_2d[:, 0].mean() - max_range/2, 
                           features_2d[:, 0].mean() + max_range/2)
        ax_features.set_ylim(features_2d[:, 1].mean() - max_range/2, 
                           features_2d[:, 1].mean() + max_range/2)
        
        # Plot hidden vectors
        ax_hidden.scatter(hidden_2d[:, 0], hidden_2d[:, 1], 
                        c='red', alpha=0.5, s=5)  # Increased point size
        ax_hidden.set_title(f'Hidden vectors (n={size})')
        
        # Set equal aspect ratio and proper limits for hidden vectors
        max_range = max(np.ptp(hidden_2d[:, 0]), np.ptp(hidden_2d[:, 1]))
        ax_hidden.set_aspect('equal')
        ax_hidden.set_xlim(hidden_2d[:, 0].mean() - max_range/2, 
                          hidden_2d[:, 0].mean() + max_range/2)
        ax_hidden.set_ylim(hidden_2d[:, 1].mean() - max_range/2, 
                          hidden_2d[:, 1].mean() + max_range/2)
        
        # Remove ticks for cleaner look
        ax_features.tick_params(axis='both', which='both', length=0)
        ax_hidden.tick_params(axis='both', which='both', length=0)
        
        # Add grid
        ax_features.grid(True, alpha=0.3, linestyle='--')
        ax_hidden.grid(True, alpha=0.3, linestyle='--')
        
        # Set background color
        ax_features.set_facecolor('white')
        ax_hidden.set_facecolor('white')
    
    plt.tight_layout()
    #plt.savefig(save_dir / f'double_descent_polytopes_{model_type}.png', 
    #            dpi=300, bbox_inches='tight', facecolor='white')
    #plt.close()

def visualize_embedding_evolution(embeddings_evolution, labels, save_dir):
    """Visualize embedding evolution horizontally with class-based coloring"""
    # Select specific sizes to visualize
    selected_sizes = [75,200, 400, 1000]
    n_rows = len(selected_sizes)  # One row per dataset size
    n_cols = 3  # Initial, Hidden, Final
    
    # Create figure with more explicit spacing
    fig = plt.figure(figsize=(20, 5*n_rows))
    
    # Create GridSpec with explicit spacing parameters
    gs = gridspec.GridSpec(n_rows, n_cols, figure=fig,
                          hspace=0.4,    # Horizontal space between rows
                          wspace=0.3,    # Vertical space between columns
                          top=0.95,      # Top margin
                          bottom=0.05,   # Bottom margin
                          left=0.1,      # Left margin
                          right=0.9)     # Right margin
    
    # Fix deprecated get_cmap
    class_cmap = plt.colormaps['tab10']
    
    # Stage labels
    stages = ['Input Features', 'Hidden Representations', 'Output Embeddings']
    
    for row, size in enumerate(selected_sizes):
        embeddings = embeddings_evolution[size]
        current_labels = labels[size]
        
        for col, (stage, stage_name) in enumerate(zip(['initial', 'hidden', 'final'], stages)):
            ax = fig.add_subplot(gs[row, col])
            
            # Get embeddings for current stage
            emb = embeddings[stage]
            
            # Process embeddings appropriately
            if stage == 'initial':
                # Initial features might be high dimensional
                emb_2d = PCA(n_components=2).fit_transform(emb.detach().cpu().numpy())
            elif stage == 'hidden':
                # Hidden layer could be any dimension
                emb_2d = PCA(n_components=2).fit_transform(emb.detach().cpu().numpy())
            else:  # final
                # Final layer could also be any dimension
                emb_2d = PCA(n_components=2).fit_transform(emb.detach().cpu().numpy())
            
            # Plot with class-based coloring
            unique_labels = np.unique(current_labels)
            for label in unique_labels:
                mask = current_labels == label
                ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
                          c=[class_cmap(label)],
                          label=f'Class {label}',
                          alpha=0.6,
                          s=50,
                          edgecolor='white',
                          linewidth=0.5)
            
            # Styling
            ax.grid(True, linestyle='--', alpha=0.3, color='gray')
            ax.set_facecolor('#f8f9fa')
            ax.tick_params(axis='both', which='both', length=0)
            
            # Box styling
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color('gray')
            
            # Titles
            if row == 0:
                ax.set_title(stage_name, fontsize=16, fontweight='bold', pad=10)
            
            # Dataset size label on the left
            if col == 0:
                ax.text(-0.2, 0.5, f'n={size}', 
                       transform=ax.transAxes,
                       fontsize=16,
                       fontweight='bold',
                       ha='right',
                       va='center')
            
            # Legend only for first row
            if row == 0 and col == n_cols-1:
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Set equal aspect ratio
            ax.set_aspect('equal')
    
    # Main title with adjusted position
    plt.suptitle('Evolution of Node Representations Through Network Layers',
                y=0.98,  # Adjusted y position
                fontsize=20,
                fontweight='bold')
    
    # Remove tight_layout call since we're using GridSpec parameters
    
    # Save figure
    plt.savefig(save_dir / 'embedding_evolution.png',
                dpi=300,
                bbox_inches='tight',
                facecolor='white')
    plt.close()

def track_node_embeddings(models, all_data, sizes, device, save_dir):
    """Track evolution of node embeddings across dataset sizes"""
    logging.info("\nTRACKING NODE EMBEDDINGS")
    logging.info("="*50)
    
    embeddings_evolution = {}
    labels = {}  # Store labels for each dataset size
    transition_sizes = [30, 75, 100, 200, 400, 1000]  # Customize this list as needed
    
    for size in transition_sizes:
        logging.info(f"\nProcessing dataset size: {size}")
        size_idx = sizes.index(size)
        model = models[size_idx]
        data = all_data[size_idx].to(device)
        
        model.eval()
        with torch.no_grad():
            initial_embedding = data.x
            hidden_embedding = model.conv1(data.x, data.edge_index)
            hidden_embedding = F.relu(hidden_embedding)
            final_embedding = model.conv2(hidden_embedding, data.edge_index)
            
            embeddings_evolution[size] = {
                'initial': initial_embedding.cpu(),
                'hidden': hidden_embedding.cpu(),
                'final': final_embedding.cpu()
            }
            labels[size] = data.y.cpu().numpy()  # Store labels
        
        logging.info(f"Captured embeddings for size {size}")
    
    logging.info("\nVisualizing embedding evolution...")
    visualize_embedding_evolution(embeddings_evolution, labels, save_dir)
    return embeddings_evolution



def select_representative_nodes(data, k=5):
    """Select representative nodes based on degree centrality"""
    edge_index = data.edge_index.cpu()
    degrees = torch.bincount(edge_index[0])
    
    # Get top-k nodes by degree
    top_k_nodes = torch.argsort(degrees, descending=True)[:k]
    return top_k_nodes.tolist()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Superposition Analysis')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default="gcn")
    parser.add_argument('--hidden_dim', type=int, default=2)
    parser.add_argument('--num_features', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    
    # GAT-specific parameters
    parser.add_argument('--gat_heads', type=int, default=1)
    
    # Dataset parameters
    parser.add_argument('--sparsity', type=float, default=0.95)
    parser.add_argument('--num_blocks', type=int, default=5)
    parser.add_argument('--sizes', type=str, default="30,50,75,100,150,200,250,300,350,400,500,1000")
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    
    # Visualization parameters
    parser.add_argument('--small_threshold', type=int, default=100)
    parser.add_argument('--large_threshold', type=int, default=400)
    parser.add_argument('--selected_sizes', type=str, default="30,50,75")
    
    # Save directory
    parser.add_argument('--save_dir', type=str, default='Explainer/PolysemanticNeurons')
    
    args = parser.parse_args()
    
    # Convert string lists to actual lists
    args.sizes = [int(x) for x in args.sizes.split(',')]
    args.selected_sizes = [int(x) for x in args.selected_sizes.split(',')]
    
    return vars(args)


def analyze_hidden_representations(models, all_data, sizes, device, save_dir):
    """Analyze how hidden representations evolve and what they learn"""
    logging.info("\nANALYZING HIDDEN REPRESENTATION STRUCTURE")
    logging.info("="*50)
    
    for size_idx, size in enumerate(sizes):
        if size not in [30, 100, 200, 1000]:
            continue
            
        model = models[size_idx]
        data = all_data[size_idx].to(device)
        
        model.eval()
        with torch.no_grad():
            # Get hidden representations
            hidden = model.conv1(data.x, data.edge_index)
            hidden = F.relu(hidden)
            hidden_np = hidden.cpu().numpy()
            
            # Get node labels and community assignments
            labels = data.y.cpu().numpy()
            
            # Create figure with multiple visualization methods
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f'Hidden Space Analysis (n={size})', fontsize=16)
            
            # 1. PCA visualization colored by class
            pca = PCA(n_components=2)
            hidden_2d_pca = pca.fit_transform(hidden_np)
            
            # Plot points colored by class label
            scatter = axes[0,0].scatter(hidden_2d_pca[:, 0], hidden_2d_pca[:, 1], 
                                      c=labels,
                                      cmap='coolwarm',
                                      alpha=0.6)
            axes[0,0].set_title('PCA Projection (colored by class)')
            axes[0,0].grid(True, alpha=0.3)
            plt.colorbar(scatter, ax=axes[0,0])
            
            # 2. t-SNE visualization colored by class
            perplexity = min(30, size - 1)
            try:
                tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                hidden_2d_tsne = tsne.fit_transform(hidden_np)
                scatter = axes[0,1].scatter(hidden_2d_tsne[:, 0], hidden_2d_tsne[:, 1],
                                          c=labels,
                                          cmap='coolwarm',
                                          alpha=0.6)
                axes[0,1].set_title(f't-SNE Projection\n(colored by class)')
                axes[0,1].grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=axes[0,1])
            except Exception as e:
                logging.warning(f"t-SNE failed for size {size}: {str(e)}")
                axes[0,1].text(0.5, 0.5, 't-SNE failed', ha='center', va='center')
            
            # 3. Analysis of class separation
            class_0_hidden = hidden_np[labels == 0]
            class_1_hidden = hidden_np[labels == 1]
            
            # Calculate mean vectors for each class
            mean_0 = np.mean(class_0_hidden, axis=0)
            mean_1 = np.mean(class_1_hidden, axis=0)
            
            # Calculate angle between class means
            cos_angle = np.dot(mean_0, mean_1) / (np.linalg.norm(mean_0) * np.linalg.norm(mean_1))
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Plot angle distribution within each class
            angles_0 = np.arccos(np.clip(np.dot(class_0_hidden, class_0_hidden.T), -1, 1))
            angles_1 = np.arccos(np.clip(np.dot(class_1_hidden, class_1_hidden.T), -1, 1))
            
            axes[1,0].hist(angles_0.flatten(), bins=50, alpha=0.5, label='Class 0')
            axes[1,0].hist(angles_1.flatten(), bins=50, alpha=0.5, label='Class 1')
            axes[1,0].set_title('Distribution of Angles within Classes')
            axes[1,0].set_xlabel('Angle (radians)')
            axes[1,0].set_ylabel('Count')
            axes[1,0].legend()
            
            # 4. Magnitude distribution by class
            magnitudes_0 = np.linalg.norm(class_0_hidden, axis=1)
            magnitudes_1 = np.linalg.norm(class_1_hidden, axis=1)
            
            axes[1,1].hist(magnitudes_0, bins=50, alpha=0.5, label='Class 0')
            axes[1,1].hist(magnitudes_1, bins=50, alpha=0.5, label='Class 1')
            axes[1,1].set_title('Distribution of Magnitudes by Class')
            axes[1,1].set_xlabel('Magnitude')
            axes[1,1].set_ylabel('Count')
            axes[1,1].legend()
            
            plt.tight_layout()
            plt.savefig(save_dir / f'hidden_space_analysis_{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log analysis
            logging.info(f"\nDataset Size: {size}")
            logging.info("-" * 30)
            logging.info(f"PCA explained variance ratios: {pca.explained_variance_ratio_}")
            logging.info(f"Angle between class means: {angle:.3f} radians")
            logging.info(f"Mean magnitude class 0: {np.mean(magnitudes_0):.3f}")
            logging.info(f"Mean magnitude class 1: {np.mean(magnitudes_1):.3f}")

def analyze_neuron_polysemanticity(models, all_data, sizes, device, save_dir):
    """Analyze whether neurons are responding to multiple classes (polysemanticity)"""
    logging.info("\nANALYZING NEURON POLYSEMANTICITY")
    logging.info("="*50)
    
    # Select specific dataset sizes for analysis
    analysis_sizes = [30, 100, 200, 1000]
    
    for size_idx, size in enumerate(sizes):
        if size not in analysis_sizes:
            continue
            
        model = models[size_idx]
        data = all_data[size_idx].to(device)
        
        model.eval()
        with torch.no_grad():
            # Get hidden layer activations
            hidden = model.conv1(data.x, data.edge_index)
            hidden = F.relu(hidden)
            hidden_np = hidden.cpu().numpy()
            
            # Get labels
            labels = data.y.cpu().numpy()
            unique_labels = np.unique(labels)
            num_classes = len(unique_labels)
            num_neurons = hidden_np.shape[1]
            
            # Calculate mean activation per class for each neuron
            class_activations = np.zeros((num_classes, num_neurons))
            for class_idx, label in enumerate(unique_labels):
                class_mask = labels == label
                class_activations[class_idx] = hidden_np[class_mask].mean(axis=0)
            
            # Calculate neuron selectivity
            # Higher values indicate more class-specific neurons
            selectivity = np.zeros(num_neurons)
            for neuron_idx in range(num_neurons):
                max_activation = np.max(class_activations[:, neuron_idx])
                other_activations = np.mean(class_activations[:, neuron_idx][
                    class_activations[:, neuron_idx] != max_activation
                ])
                selectivity[neuron_idx] = (max_activation - other_activations) / (max_activation + other_activations + 1e-6)
            
            # Create visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            fig.suptitle(f'Neuron Polysemanticity Analysis (n={size})', fontsize=16)
            
            # 1. Heatmap of class-neuron activations
            im = axes[0,0].imshow(class_activations, aspect='auto', cmap='coolwarm')
            axes[0,0].set_title('Mean Neuron Activation by Class')
            axes[0,0].set_xlabel('Neuron Index')
            axes[0,0].set_ylabel('Class')
            plt.colorbar(im, ax=axes[0,0])
            
            # 2. Neuron selectivity distribution
            axes[0,1].hist(selectivity, bins=20)
            axes[0,1].set_title('Distribution of Neuron Selectivity')
            axes[0,1].set_xlabel('Selectivity Score')
            axes[0,1].set_ylabel('Count')
            
            # 3. Top polysemantic neurons analysis
            polysemantic_threshold = np.percentile(selectivity, 25)  # Bottom 25% are considered polysemantic
            polysemantic_neurons = np.where(selectivity < polysemantic_threshold)[0]
            
            if len(polysemantic_neurons) > 0:
                poly_activations = class_activations[:, polysemantic_neurons]
                im = axes[1,0].imshow(poly_activations, aspect='auto', cmap='coolwarm')
                axes[1,0].set_title('Top Polysemantic Neurons')
                axes[1,0].set_xlabel('Neuron Index')
                axes[1,0].set_ylabel('Class')
                plt.colorbar(im, ax=axes[1,0])
            
            # 4. Class correlation matrix
            class_corr = np.corrcoef(class_activations)
            im = axes[1,1].imshow(class_corr, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1,1].set_title('Class Activation Correlation')
            plt.colorbar(im, ax=axes[1,1])
            
            plt.tight_layout()
            plt.savefig(save_dir / f'polysemanticity_analysis_{size}.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Log analysis results
            logging.info(f"\nDataset Size: {size}")
            logging.info("-" * 30)
            logging.info(f"Mean neuron selectivity: {np.mean(selectivity):.3f}")
            logging.info(f"Number of highly polysemantic neurons: {len(polysemantic_neurons)}")
            
            # Calculate percentage of neurons that respond strongly to multiple classes
            activation_threshold = 0.5  # Consider a neuron "active" if activation > 0.5
            multi_class_neurons = 0
            for neuron_idx in range(num_neurons):
                active_classes = np.sum(class_activations[:, neuron_idx] > activation_threshold)
                if active_classes > 1:
                    multi_class_neurons += 1
            
            polysemantic_ratio = multi_class_neurons / num_neurons
            logging.info(f"Proportion of polysemantic neurons: {polysemantic_ratio:.3f}")
            
            # Calculate average activation overlap between classes
            class_overlaps = []
            for i in range(num_classes):
                for j in range(i+1, num_classes):
                    overlap = np.mean(
                        np.minimum(class_activations[i], class_activations[j]) /
                        np.maximum(class_activations[i], class_activations[j])
                    )
                    class_overlaps.append(overlap)
            
            logging.info(f"Mean class activation overlap: {np.mean(class_overlaps):.3f}")

def main():
    # Replace the config dictionary with parsed arguments
    config = parse_args()
    
    # Convert save_dir string to Path and create directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(exist_ok=True, parents=True)
    
    # Set up logging
    setup_logging(save_dir, config)
    
    # ====================== SETUP ======================
    random.seed(config['seed'])
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    all_test_losses = []
    all_models = []
    all_data = []
    results = {}  # Initialize results dictionary
    
    # ====================== TRAINING LOOP ======================
    logging.info("TRAINING PROGRESS")
    logging.info("="*80)
    
    for size in config['sizes']:
        logging.info(f"\nDataset Size: {size}")
        logging.info("-"*50)
        
        # Generate dataset
        data = generate_sbm_dataset(
            num_nodes=size,
            num_features=config['num_features'],
            num_classes=config['num_classes'],
            num_blocks=config['num_blocks'],
            sparsity=config['sparsity'],
            seed=config['seed']
        )
        
        # Add dataset statistics logging
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1) // 2  # Divide by 2 since edges are bidirectional
        avg_degree = (2 * num_edges) / num_nodes
        
        logging.info("\nDataset Statistics:")
        logging.info(f"Number of nodes: {num_nodes}")
        logging.info(f"Number of edges: {num_edges}")
        logging.info(f"Average node degree: {avg_degree:.2f}")
        logging.info("-"*50)

        # Initialize model based on type
        if config['model_type'] == "gcn":
            model = SimpleGCN(
                num_features=config['num_features'],
                num_classes=config['num_classes'],
                hidden_channels=config['hidden_dim']
            ).to(device)
        else:  # gat
            model = SimpleGAT(
                num_features=config['num_features'],
                num_classes=config['num_classes'],
                hidden_channels=config['hidden_dim'],
                heads=config['gat_heads']
            ).to(device)
            
        optimizer = torch.optim.Adam(model.parameters(), lr=config['base_lr'], weight_decay=config['weight_decay'])
        criterion = nn.CrossEntropyLoss()
        
        data = data.to(device)
        best_val_loss = float('inf')
        best_epoch = 0
        
        # Training loop
        for epoch in range(config['num_epochs']):
            model.train()
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 100 == 0:
                model.eval()
                with torch.no_grad():
                    out = model(data.x, data.edge_index)
                    val_loss = criterion(out[data.val_mask], data.y[data.val_mask])
                    test_loss = criterion(out[data.test_mask], data.y[data.test_mask])
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_test_loss = test_loss.item()
                        best_epoch = epoch + 1
                    
                    logging.info(
                        f"Epoch {epoch+1:4d}/{config['num_epochs']} | "
                        f"Train: {loss.item():.4f} | "
                        f"Val: {val_loss:.4f} | "
                        f"Test: {test_loss:.4f}"
                    )
        
        logging.info(f"\nBest Results - Epoch: {best_epoch} | Test Loss: {best_test_loss:.4f}")
        logging.info("-"*50)
        
        # After training each model, store results
        model.eval()
        with torch.no_grad():
            # Get hidden representations
            hidden_vectors = []
            
            # First layer
            x = model.conv1(data.x, data.edge_index)
            x = F.relu(x)
            hidden_vectors.append(x)
            
            # Second layer
            x = model.conv2(x, data.edge_index)
            hidden_vectors.append(x)
            
            # Store results for this dataset size
            results[size] = {
                'test_loss': best_test_loss,
                'features': data.x,
                'hidden_vectors': hidden_vectors
            }
        
        all_test_losses.append(best_test_loss)
        all_models.append(model)
        all_data.append(data)
    
    logging.info("\nTRAINING COMPLETE")
    logging.info("="*80)
    #logging.info("Results saved in 'SBM_Superposition_Results' directory\n")
    
    # ====================== VISUALIZATION ======================
    plot_double_descent_and_polytopes(
        sizes=config['sizes'],
        losses=all_test_losses,
        models=all_models,
        all_data=all_data,
        save_dir=save_dir,
        small_threshold=config['small_threshold'],
        large_threshold=config['large_threshold'],
        model_type=config['model_type'],
        device=device
    )
    
    # sparsity_values = visualize_superposition(
    #     models=all_models,
    #     all_data=all_data,
    #     sizes=config['sizes'],
    #     save_dir=save_dir,
    #     device=device,
    #     selected_sizes=config['selected_sizes'],
    #     model_type=config['model_type']
    # )
    
    # After training loop, before saving results
    for size in config['selected_sizes']:
        size_idx = config['sizes'].index(size)
        model = all_models[size_idx]
        data = all_data[size_idx].to(device)
        
        # Visualize polytope
        visualize_polytopes(
            models=all_models,
            all_data=all_data,
            sizes=config['sizes'],
            save_dir=save_dir,
            device=device,
            model_type=config['model_type']
        )
    
    # Plot double descent curve with the populated results
    plot_double_descent_curve(results, save_dir)
    
    logging.info("Analysis complete! Results saved in 'SBM_Superposition_Results' directory")
    
    # After training loop, collect results
    results = {
        'dataset_sizes': config['sizes'],
        'test_losses': all_test_losses,
        'config': config
    }
    
    # Save results
    results_df = pd.DataFrame({
        'dataset_sizes': config['sizes'],
        'test_losses': all_test_losses
    })
    results_df.to_csv(Path(config['save_dir']) / 'results.csv', index=False)
    
    # Track embedding evolution
    embeddings_evolution = track_node_embeddings(
        all_models,
        all_data,
        config['sizes'],
        device,
        save_dir
    )
    
    # Add the new analysis
    analyze_hidden_representations(
        all_models,
        all_data,
        config['sizes'],
        device,
        save_dir
    )
    
    # Add the polysemanticity analysis
    analyze_neuron_polysemanticity(
        all_models,
        all_data,
        config['sizes'],
        device,
        save_dir
    )
    
    return results

if __name__ == "__main__":
    main()