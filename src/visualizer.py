import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path


class Visualizer:
    #Generate figures for paper
    
    def __init__(self, outputDir: str = "experiments/figures/"):
        self.outputDir = Path(outputDir)
        self.outputDir.mkdir(parents=True, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
    
    def plotModelComparison(self, results: pd.DataFrame, filename: str = "model_comparison.png"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        results['rmsle'].plot(kind='barh', ax=axes[0], color=['#d62728', '#1f77b4', '#2ca02c'])
        axes[0].set_xlabel('RMSLE (lower is better)')
        axes[0].set_title('Model Accuracy')
        axes[0].invert_yaxis()
        
        if 'training_time' in results.columns:
            results['training_time'].plot(kind='barh', ax=axes[1], color='#2ca02c')
            axes[1].set_xlabel('Training Time (s)')
            axes[1].set_title('Computational Cost')
            axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.outputDir / filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plotFeatureImportance(self, df: pd.DataFrame, topN: int = 15, filename: str = "feature_importance.png"):
        fig, ax = plt.subplots(figsize=(8, 6))
        top = df.head(topN)
        ax.barh(top['feature'], top['importance'], color=plt.cm.Blues(np.linspace(0.4, 0.9, len(top)))[::-1])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {topN} Features')
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(self.outputDir / filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plotAblationStudy(self, df: pd.DataFrame, filename: str = "ablation_study.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        plotDf = df[df['removed'] != 'none'].sort_values('rmsle_increase')
        colors = ['#d62728' if x > 0 else '#2ca02c' for x in plotDf['rmsle_increase']]
        ax.barh(plotDf['removed'], plotDf['rmsle_increase'], color=colors)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('RMSLE Increase')
        ax.set_title('Ablation Study')
        plt.tight_layout()
        plt.savefig(self.outputDir / filename, bbox_inches='tight', dpi=150)
        plt.close()
    
    def plotLearningCurves(self, history: dict, filename: str = "learning_curves.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(history['train_loss']) + 1)
        ax.plot(epochs, history['train_loss'], 'b-', label='Train')
        ax.plot(epochs, history['valid_loss'], 'r-', label='Valid')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('LSTM Training')
        ax.legend()
        plt.tight_layout()
        plt.savefig(self.outputDir / filename, bbox_inches='tight', dpi=150)
        plt.close()