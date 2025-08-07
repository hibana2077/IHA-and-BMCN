import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import torch.nn.functional as F
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class MetricsCalculator:
    """Calculate comprehensive metrics for UFGVC evaluation"""
    
    def __init__(self, num_classes: int):
        self.num_classes = num_classes
    
    def calculate_all_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """Calculate all evaluation metrics"""
        # Convert to numpy
        if isinstance(outputs, torch.Tensor):
            probs = F.softmax(outputs, dim=1).numpy()
            predictions = outputs.argmax(dim=1).numpy()
        else:
            probs = outputs
            predictions = outputs.argmax(axis=1)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        
        metrics = {}
        
        # Basic accuracy metrics
        metrics.update(self._calculate_accuracy_metrics(probs, targets))
        
        # Fine-grained metrics
        metrics.update(self._calculate_fine_grained_metrics(predictions, targets))
        
        # Embedding quality metrics (using predicted probabilities as features)
        metrics.update(self._calculate_embedding_metrics(probs, targets))
        
        # Margin analysis
        metrics.update(self._calculate_margin_metrics(probs, targets))
        
        # Calibration metrics
        metrics.update(self._calculate_calibration_metrics(probs, targets))
        
        return metrics
    
    def _calculate_accuracy_metrics(self, probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate Top-1 and Top-5 accuracy"""
        predictions = probs.argmax(axis=1)
        top1_acc = 100.0 * np.mean(predictions == targets)
        
        # Top-5 accuracy
        top5_predictions = probs.argsort(axis=1)[:, -5:]
        top5_acc = 100.0 * np.mean([targets[i] in top5_predictions[i] for i in range(len(targets))])
        
        return {
            'top1_acc': top1_acc,
            'top5_acc': top5_acc
        }
    
    def _calculate_fine_grained_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate class-wise F1 and macro recall"""
        # Per-class F1 scores
        f1_scores = f1_score(targets, predictions, average=None, zero_division=0)
        macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)
        weighted_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
        
        # Confusion matrix for per-class recall
        cm = confusion_matrix(targets, predictions, labels=list(range(self.num_classes)))
        per_class_recall = np.diag(cm) / (cm.sum(axis=1) + 1e-8)
        macro_recall = np.mean(per_class_recall)
        
        return {
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'macro_recall': macro_recall,
            'min_class_f1': np.min(f1_scores),
            'max_class_f1': np.max(f1_scores),
            'std_class_f1': np.std(f1_scores)
        }
    
    def _calculate_embedding_metrics(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate NMI and ARI on feature space using clustering"""
        # Use predicted probabilities as features for clustering
        if len(np.unique(targets)) < 2:
            return {'nmi': 0.0, 'ari': 0.0}
        
        try:
            # Check if features are too similar (which causes clustering warnings)
            feature_std = np.std(features, axis=0)
            if np.all(feature_std < 1e-6):  # All features are nearly identical
                return {'nmi': 0.0, 'ari': 0.0}
            
            # Perform K-means clustering
            n_clusters = min(len(np.unique(targets)), len(features))  # Ensure n_clusters <= n_samples
            if n_clusters < 2:
                return {'nmi': 0.0, 'ari': 0.0}
                
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10, max_iter=100)
            cluster_labels = kmeans.fit_predict(features)
            
            # Calculate clustering metrics only if we have valid clusters
            if len(np.unique(cluster_labels)) > 1:
                nmi = normalized_mutual_info_score(targets, cluster_labels)
                ari = adjusted_rand_score(targets, cluster_labels)
            else:
                nmi, ari = 0.0, 0.0
            
            return {
                'nmi': nmi,
                'ari': ari
            }
        except Exception:
            return {'nmi': 0.0, 'ari': 0.0}
    
    def _calculate_margin_metrics(self, probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate inter/intra-class cosine distances"""
        # Normalize features for cosine distance
        features_norm = probs / (np.linalg.norm(probs, axis=1, keepdims=True) + 1e-8)
        
        # Calculate pairwise cosine similarities
        similarities = np.dot(features_norm, features_norm.T)
        
        # Intra-class similarities (same class)
        intra_similarities = []
        for class_id in range(self.num_classes):
            class_mask = targets == class_id
            if np.sum(class_mask) > 1:  # Need at least 2 samples
                class_indices = np.where(class_mask)[0]
                class_sims = similarities[np.ix_(class_indices, class_indices)]
                # Remove diagonal (self-similarity)
                mask = ~np.eye(class_sims.shape[0], dtype=bool)
                intra_similarities.extend(class_sims[mask])
        
        # Inter-class similarities (different classes)
        inter_similarities = []
        for i in range(len(targets)):
            for j in range(i + 1, len(targets)):
                if targets[i] != targets[j]:
                    inter_similarities.append(similarities[i, j])
        
        # Convert to distances (1 - cosine similarity)
        intra_distances = [1 - sim for sim in intra_similarities] if intra_similarities else [0]
        inter_distances = [1 - sim for sim in inter_similarities] if inter_similarities else [1]
        
        return {
            'mean_intra_distance': np.mean(intra_distances),
            'mean_inter_distance': np.mean(inter_distances),
            'margin': np.mean(inter_distances) - np.mean(intra_distances)
        }
    
    def _calculate_calibration_metrics(self, probs: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """Calculate Expected Calibration Error (ECE)"""
        predictions = probs.argmax(axis=1)
        confidences = probs.max(axis=1)
        accuracies = (predictions == targets).astype(float)
        
        # ECE calculation with 10 bins
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = accuracies[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            'ece': ece,
            'avg_confidence': np.mean(confidences),
            'avg_accuracy': np.mean(accuracies)
        }
    
    def save_confusion_matrix(self, outputs: torch.Tensor, targets: torch.Tensor, 
                            save_path: Path, class_names=None):
        """Save confusion matrix as PNG and CSV"""
        if isinstance(outputs, torch.Tensor):
            predictions = outputs.argmax(dim=1).numpy()
        else:
            predictions = outputs.argmax(axis=1)
        
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()
        
        # Create confusion matrix
        cm = confusion_matrix(targets, predictions, labels=list(range(self.num_classes)))
        
        # Save as CSV
        np.savetxt(save_path.with_suffix('.csv'), cm, delimiter=',', fmt='%d')
        
        # Create and save plot
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names[:self.num_classes] if class_names else None,
                   yticklabels=class_names[:self.num_classes] if class_names else None)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig(save_path.with_suffix('.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def calculate_efficiency_metrics(self, model: torch.nn.Module, 
                                   input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> Dict[str, Any]:
        """Calculate model efficiency metrics"""
        device = next(model.parameters()).device
        dummy_input = torch.randn(input_shape).to(device)
        
        # Parameter count
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Memory usage (approximate)
        model.eval()
        with torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            _ = model(dummy_input)
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'peak_memory_mb': peak_memory
        }
