import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px

class TrainingVisualizer:
    def __init__(self, save_dir='./visualizations'):
        self.save_dir = save_dir
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'learning_rates': []
        }
    
    def update_history(self, train_loss, val_loss, train_acc, val_acc, lr=None):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
        if lr is not None:
            self.history['learning_rates'].append(lr)
    
    def plot_training_curves(self, filename='training_curves.png'):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Training Loss')
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.history['train_acc'], 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, cm, class_names, filename='confusion_matrix.png'):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self, feature_importance, feature_names, top_k=20, filename='feature_importance.png'):
        indices = np.argsort(feature_importance)[-top_k:]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(indices)), feature_importance[indices])
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.title(f'Top {top_k} Feature Importance')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

class FeatureVisualizer:
    def __init__(self, save_dir='./visualizations'):
        self.save_dir = save_dir
    
    def plot_tsne(self, features, labels, class_names, filename='tsne_plot.png'):
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = tsne.fit_transform(features)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                            cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('t-SNE Visualization of Features')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        
        if class_names is not None:
            legend_elements = []
            for i, class_name in enumerate(class_names):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=plt.cm.tab10(i/len(class_names)),
                                                markersize=8, label=class_name))
            plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_pca(self, features, labels, class_names, filename='pca_plot.png'):
        pca = PCA(n_components=2, random_state=42)
        features_2d = pca.fit_transform(features)
        
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, 
                            cmap='tab10', alpha=0.7)
        plt.colorbar(scatter)
        plt.title(f'PCA Visualization (Explained Variance: {pca.explained_variance_ratio_.sum():.2f})')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
        
        if class_names is not None:
            legend_elements = []
            for i, class_name in enumerate(class_names):
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=plt.cm.tab10(i/len(class_names)),
                                                markersize=8, label=class_name))
            plt.legend(handles=legend_elements, loc='best')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_distributions(self, features, feature_names, labels, class_names, 
                                 n_features=9, filename='feature_distributions.png'):
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten()
        
        selected_features = np.random.choice(len(feature_names), n_features, replace=False)
        
        for i, feature_idx in enumerate(selected_features):
            if i >= len(axes):
                break
                
            for class_idx in range(len(class_names)):
                class_mask = labels == class_idx
                axes[i].hist(features[class_mask, feature_idx], alpha=0.7, 
                           label=class_names[class_idx], bins=20)
            
            axes[i].set_title(f'{feature_names[feature_idx]}')
            axes[i].legend()
        
        for i in range(len(selected_features), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()

class ModelComparisonVisualizer:
    def __init__(self, save_dir='./visualizations'):
        self.save_dir = save_dir
    
    def plot_model_comparison(self, results_df, metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                            filename='model_comparison.png'):
        n_metrics = len(metrics)
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            if i >= len(axes):
                break
                
            model_names = results_df['model'].unique()
            metric_values = []
            metric_stds = []
            
            for model in model_names:
                model_data = results_df[results_df['model'] == model]
                metric_values.append(model_data[f'{metric}_mean'].values[0])
                metric_stds.append(model_data[f'{metric}_std'].values[0])
            
            bars = axes[i].bar(model_names, metric_values, yerr=metric_stds, 
                             capsize=5, alpha=0.7, color=plt.cm.Set3(range(len(model_names))))
            axes[i].set_title(f'{metric.capitalize()} Comparison')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].tick_params(axis='x', rotation=45)
            
            for bar, value in zip(bars, metric_values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/{filename}', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_radar_chart(self, results_df, models_to_compare, metrics=['accuracy', 'precision', 'recall', 'f1_score'],
                        filename='radar_chart.png'):
        fig = go.Figure()
        
        for model in models_to_compare:
            model_data = results_df[results_df['model'] == model].iloc[0]
            values = [model_data[f'{metric}_mean'] for metric in metrics]
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metrics,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='Model Performance Radar Chart'
        )
        
        fig.write_image(f'{self.save_dir}/{filename}')
    
    def create_performance_table(self, results_df, filename='performance_table.html'):
        table_data = []
        
        for _, row in results_df.iterrows():
            table_data.append({
                'Model': row['model'],
                'Accuracy': f"{row['accuracy_mean']:.3f} ± {row['accuracy_std']:.3f}",
                'Precision': f"{row['precision_mean']:.3f} ± {row['precision_std']:.3f}",
                'Recall': f"{row['recall_mean']:.3f} ± {row['recall_std']:.3f}",
                'F1-Score': f"{row['f1_score_mean']:.3f} ± {row['f1_score_std']:.3f}",
                'Inference Time (ms)': f"{row['inference_time_mean']:.2f} ± {row['inference_time_std']:.2f}",
                'Parameters (M)': f"{row['parameters_millions']:.2f}" if 'parameters_millions' in row else 'N/A'
            })
        
        df_table = pd.DataFrame(table_data)
        html_table = df_table.to_html(index=False, classes='table table-striped', border=0)
        
        with open(f'{self.save_dir}/{filename}', 'w') as f:
            f.write(html_table)
        
        return df_table

def create_visualization_suite(config):
    save_dir = config.get('visualization_save_dir', './visualizations')
    return {
        'training': TrainingVisualizer(save_dir),
        'features': FeatureVisualizer(save_dir),
        'comparison': ModelComparisonVisualizer(save_dir)
    }