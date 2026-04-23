"""
Visualization Suite for Bio-Music Pipeline.

Provides scientific-quality visualizations for research papers:
- PCA / t-SNE / UMAP of bio-vectors
- Correlation heatmaps (bio-features ↔ musical parameters)
- Attention maps from BioConditioningModule
- Piano roll renderings of generated music
- Distribution comparisons across conditions
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json


def visualize_bio_vectors(bio_vectors: np.ndarray,
                           labels: np.ndarray = None,
                           output_dir: str = None,
                           methods: List[str] = None) -> Dict[str, str]:
    """
    Create dimensionality reduction visualizations of bio-vectors.

    Args:
        bio_vectors: Array of shape (n_samples, 128)
        labels: Optional labels for coloring (e.g., organism names)
        output_dir: Directory to save figures
        methods: List of methods ('pca', 'tsne', 'umap')

    Returns:
        Dictionary mapping method to saved figure path
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if output_dir is None:
        output_dir = Path('results/visualizations')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if methods is None:
        methods = ['pca', 'tsne']

    # Try UMAP if requested
    has_umap = False
    if 'umap' in methods:
        try:
            import umap
            has_umap = True
        except ImportError:
            print("Warning: UMAP not installed. Install with: pip install umap-learn")

    results = {}

    # ============ PCA ============
    if 'pca' in methods:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        embedding = pca.fit_transform(bio_vectors)

        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[colors[i]], label=str(label), alpha=0.7, s=50)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)

        ax.set_title(f'PCA of Bio-Vectors\n'
                     f'PC1: {pca.explained_variance_ratio_[0]:.1%}, '
                     f'PC2: {pca.explained_variance_ratio_[1]:.1%}',
                     fontsize=14)
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        pca_path = Path(output_dir) / 'bio_vectors_pca.png' if output_dir else None
        if pca_path:
            fig.savefig(pca_path, dpi=150, bbox_inches='tight')
            results['pca'] = str(pca_path)
        plt.close(fig)

    # ============ t-SNE ============
    if 'tsne' in methods:
        if len(bio_vectors) < 3:
            print("Skipping t-SNE: requires at least 3 samples")
        else:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE

            # Use PCA first to reduce dimensionality (faster t-SNE)
            if bio_vectors.shape[1] > 50:
                pca_init = PCA(n_components=50, random_state=42)
                bio_reduced = pca_init.fit_transform(bio_vectors)
            else:
                bio_reduced = bio_vectors

            perplexity = min(30, max(2, len(bio_vectors) // 3))
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            embedding = tsne.fit_transform(bio_reduced)

            fig, ax = plt.subplots(figsize=(10, 8))

            if labels is not None:
                unique_labels = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
                for i, label in enumerate(unique_labels):
                    mask = labels == label
                    ax.scatter(embedding[mask, 0], embedding[mask, 1],
                               c=[colors[i]], label=str(label), alpha=0.7, s=50)
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)

            ax.set_title('t-SNE of Bio-Vectors', fontsize=14)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()

            tsne_path = Path(output_dir) / 'bio_vectors_tsne.png' if output_dir else None
            if tsne_path:
                fig.savefig(tsne_path, dpi=150, bbox_inches='tight')
                results['tsne'] = str(tsne_path)
            plt.close(fig)

    # ============ UMAP ============
    if 'umap' in methods and has_umap:
        import umap

        n_neighbors = min(15, max(2, len(bio_vectors) - 1))
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
        embedding = reducer.fit_transform(bio_vectors)

        fig, ax = plt.subplots(figsize=(10, 8))

        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax.scatter(embedding[mask, 0], embedding[mask, 1],
                          c=[colors[i]], label=str(label), alpha=0.7, s=50)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            ax.scatter(embedding[:, 0], embedding[:, 1], alpha=0.7, s=50)

        ax.set_title('UMAP of Bio-Vectors', fontsize=14)
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        umap_path = Path(output_dir) / 'bio_vectors_umap.png' if output_dir else None
        if umap_path:
            fig.savefig(umap_path, dpi=150, bbox_inches='tight')
            results['umap'] = str(umap_path)
        plt.close(fig)

    return results


def plot_correlation_heatmap(bio_features: np.ndarray,
                              musical_params: Dict[str, np.ndarray],
                              output_dir: str = None,
                              method: str = 'spearman') -> str:
    """
    Create correlation heatmap between bio-features and musical parameters.

    Args:
        bio_features: Array of shape (n_samples, n_bio_features)
        musical_params: Dictionary mapping param name to array (n_samples,)
        output_dir: Directory to save figure
        method: Correlation method ('pearson', 'spearman')

    Returns:
        Path to saved figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy import stats

    # Build correlation matrix
    param_names = list(musical_params.keys())
    n_bio = bio_features.shape[1]
    n_music = len(param_names)

    corr_matrix = np.zeros((n_bio, n_music))
    p_value_matrix = np.zeros((n_bio, n_music))

    for i in range(n_bio):
        for j, param_name in enumerate(param_names):
            x = bio_features[:, i]
            y = musical_params[param_name]

            if method == 'pearson':
                corr, p = stats.pearsonr(x, y)
            else:
                corr, p = stats.spearmanr(x, y)

            corr_matrix[i, j] = corr
            p_value_matrix[i, j] = p

    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, n_bio * 0.3)))

    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add significance markers
    for i in range(n_bio):
        for j in range(n_music):
            if p_value_matrix[i, j] < 0.05:
                ax.plot(j, i, marker='*', markersize=4, color='black')

    ax.set_xticks(range(n_music))
    ax.set_xticklabels(param_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_bio))
    ax.set_yticklabels([f'Bio-{i}' for i in range(n_bio)], fontsize=7)
    ax.set_xlabel('Musical Parameters', fontsize=12)
    ax.set_ylabel('Bio-Vector Components', fontsize=12)
    ax.set_title(f'Correlation Heatmap ({method.capitalize()})\n'
                 f'* = p < 0.05', fontsize=14)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05)
    cbar.set_label('Correlation Coefficient', fontsize=10)

    plt.tight_layout()

    heatmap_path = Path(output_dir) / 'correlation_heatmap.png' if output_dir else None
    if heatmap_path:
        fig.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return str(heatmap_path) if heatmap_path else None


def render_piano_roll(tokens: np.ndarray,
                       vocab: Dict[str, int],
                       output_path: str = None,
                       title: str = None,
                       max_notes: int = 200) -> str:
    """
    Render a piano roll visualization of generated music.

    Args:
        tokens: Token sequence (list of int)
        vocab: Token vocabulary
        output_path: Path to save figure
        title: Figure title
        max_notes: Maximum number of notes to display

    Returns:
        Path to saved figure
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Parse tokens into note events
    idx_to_token = {v: k for k, v in vocab.items()}

    notes = []  # (start_time, pitch, duration, velocity)
    current_time = 0
    active_notes = {}  # pitch -> (start_time, velocity)
    current_velocity = 64

    for token_id in tokens:
        if token_id not in idx_to_token:
            continue

        token = idx_to_token[token_id]

        if token.startswith('SHIFT_'):
            shift = int(token.split('_')[1])
            current_time += shift

        elif token.startswith('VEL_'):
            current_velocity = int(token.split('_')[1])

        elif token.startswith('NOTE_ON_'):
            pitch = int(token.split('_')[2])
            active_notes[pitch] = (current_time, current_velocity)

        elif token.startswith('NOTE_OFF_'):
            pitch = int(token.split('_')[2])
            if pitch in active_notes:
                start_time, velocity = active_notes.pop(pitch)
                notes.append((start_time, pitch, current_time - start_time, velocity))

        if len(notes) >= max_notes:
            break

    if not notes:
        return None

    # Render piano roll
    fig, ax = plt.subplots(figsize=(16, 6))

    # Group notes by pitch class (C, C#, D, ...)
    pitch_classes = set()
    for _, pitch, _, _ in notes:
        pitch_classes.add(pitch % 12)

    # Draw notes
    for start_time, pitch, duration, velocity in notes:
        # Color by velocity
        color_intensity = velocity / 127.0
        color = (color_intensity, 0.3, 1 - color_intensity)

        rect = patches.Rectangle((start_time, pitch), duration, 0.8,
                                linewidth=0.5, edgecolor='white',
                                facecolor=color, alpha=0.8)
        ax.add_patch(rect)

    ax.set_xlim(0, max(start_time + duration for start_time, _, duration, _ in notes) * 1.05)
    ax.set_ylim(min(pitch for _, pitch, _, _ in notes) - 2,
                max(pitch for _, pitch, _, _ in notes) + 2)

    ax.set_xlabel('Time (ticks)', fontsize=12)
    ax.set_ylabel('MIDI Pitch', fontsize=12)
    if title:
        ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def visualize_attention(model,
                        bio_vector: np.ndarray,
                        device: torch.device,
                        output_dir: str = None) -> Dict:
    """
    Visualize how BioConditioningModule processes bio-vector.

    This creates a feature importance map showing which bio-vector
    dimensions have the largest impact on the conditioned hidden states.

    Args:
        model: Trained model with bio_conditioning module
        bio_vector: Single bio-vector (128 dim)
        device: Device to run on
        output_dir: Directory to save figures

    Returns:
        Dictionary with visualization paths
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    model.eval()

    bio_tensor = torch.tensor(bio_vector, dtype=torch.float32).unsqueeze(0).to(device)

    # Create dummy hidden state
    seq_len = 10
    d_model = model.d_model
    hidden = torch.randn(seq_len, 1, d_model).to(device)

    # Get conditioned output
    with torch.no_grad():
        conditioned = model.bio_conditioning(hidden, bio_tensor)

    # Compute feature importance via gradient-based saliency
    hidden.requires_grad_(True)
    bio_tensor.requires_grad_(True)

    conditioned = model.bio_conditioning(hidden, bio_tensor)
    # Sum over all outputs and backprop
    conditioned.sum().backward()

    # Get gradients w.r.t. bio_vector
    if bio_tensor.grad is not None:
        importance = bio_tensor.grad.abs().squeeze(0).cpu().numpy()

        fig, ax = plt.subplots(figsize=(14, 5))

        # Create bar plot
        x = np.arange(len(importance))
        ax.bar(x, importance, color='steelblue', alpha=0.8)

        # Add component labels
        from ..evaluation.ablation import BIO_VECTOR_COMPONENTS
        for name, (start, end) in BIO_VECTOR_COMPONENTS.items():
            if end <= len(importance):
                ax.axvspan(start - 0.5, end - 0.5, alpha=0.1, color='red')
                ax.text((start + end) / 2, max(importance) * 0.9,
                       name, ha='center', fontsize=7, rotation=90)

        ax.set_xlabel('Bio-Vector Dimension', fontsize=12)
        ax.set_ylabel('Gradient Magnitude (Importance)', fontsize=12)
        ax.set_title('Bio-Vector Feature Importance\n'
                     '(Gradient-based Saliency)', fontsize=14)
        ax.set_xlim(0, len(importance))
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        importance_path = Path(output_dir) / 'bio_feature_importance.png' if output_dir else None
        if importance_path:
            fig.savefig(importance_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        return {
            'importance_path': str(importance_path) if importance_path else None,
            'importance_values': importance.tolist()
        }

    return {}


def create_all_visualizations(bio_vectors: np.ndarray,
                               generated_samples: Dict[str, np.ndarray],
                               vocab: Dict[str, int],
                               musical_params_dict: Dict = None,
                               model=None,
                               device=None,
                               output_dir: str = None) -> Dict:
    """
    Create all visualizations in one call.

    Args:
        bio_vectors: Bio-vectors array
        generated_samples: Dict of generated sequences by condition
        vocab: Token vocabulary
        musical_params_dict: Optional dict of musical parameters
        model: Optional trained model
        device: Optional device
        output_dir: Output directory

    Returns:
        Dictionary with all visualization paths
    """
    if output_dir is None:
        output_dir = Path('results/visualizations')
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Bio-vector visualizations
    print("Creating PCA visualization...")
    pca_results = visualize_bio_vectors(bio_vectors, output_dir=str(output_dir))
    results['bio_vector_plots'] = pca_results

    # 2. Correlation heatmap (if musical params available)
    if musical_params_dict:
        print("Creating correlation heatmap...")
        bio_features = bio_vectors[:, :95]  # First 95 are raw features
        heatmap_path = plot_correlation_heatmap(
            bio_features, musical_params_dict, output_dir=str(output_dir)
        )
        results['correlation_heatmap'] = heatmap_path

    # 3. Piano rolls for generated samples
    if generated_samples and vocab:
        print("Creating piano rolls...")
        piano_rolls = {}
        for condition, sequences in generated_samples.items():
            # Render first 5 samples per condition
            condition_paths = []
            for i in range(min(5, len(sequences))):
                tokens = sequences[i]
                piano_path = output_dir / f'piano_roll_{condition}_{i}.png'
                render_piano_roll(tokens, vocab, str(piano_path),
                                 title=f'{condition} - Sample {i}')
                condition_paths.append(str(piano_path))
            piano_rolls[condition] = condition_paths
        results['piano_rolls'] = piano_rolls

    # 4. Feature importance (if model available)
    if model and device and output_dir:
        print("Creating feature importance visualization...")
        importance_results = visualize_attention(
            model, bio_vectors[0], device, str(output_dir)
        )
        results['feature_importance'] = importance_results

    print(f"\nAll visualizations saved to: {output_dir}")
    return results
