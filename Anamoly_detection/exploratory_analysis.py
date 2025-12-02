"""
Exploratory Data Analysis Module
Generates comprehensive visualizations and statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import config

# Set plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def generate_eda_report(df=None):
    """
    Generate comprehensive EDA visualizations and report
    
    Args:
        df: DataFrame to analyze (if None, loads from dataset)
    """
    
    print("=" * 60)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 60)
    
    # Load data if not provided
    if df is None:
        print(f"\n[1/8] Loading dataset...")
        df = pd.read_csv(config.DATASET_PATH)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        print(f"   ✓ Loaded {len(df)} rows")
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'device_id', 'patient_id', 'health_event']]
    
    # 1. Distribution plots for all vital signs
    print("\n[2/8] Generating distribution plots...")
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        if idx < len(axes):
            axes[idx].hist(df[col], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{col.replace("_", " ").title()} Distribution', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col.replace("_", " ").title())
            axes[idx].set_ylabel('Frequency')
            axes[idx].grid(alpha=0.3)
            
            # Add mean and median lines
            mean_val = df[col].mean()
            median_val = df[col].median()
            axes[idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
            axes[idx].axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
            axes[idx].legend(fontsize=8)
    
    # Hide unused subplots
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(config.EDA_DIR / 'feature_distributions.png', dpi=config.DPI, bbox_inches='tight')
    print(f"   ✓ Saved: feature_distributions.png")
    plt.close()
    
    # 2. Correlation heatmap
    print("\n[3/8] Generating correlation heatmap...")
    plt.figure(figsize=(14, 12))
    correlation_matrix = df[feature_cols].corr()
    
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', 
                cmap='coolwarm', center=0, square=True, linewidths=1,
                cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(config.EDA_DIR / 'correlation_heatmap.png', dpi=config.DPI, bbox_inches='tight')
    print(f"   ✓ Saved: correlation_heatmap.png")
    plt.close()
    
    # 3. Time-series plots for key vitals
    print("\n[4/8] Generating time-series plots...")
    key_vitals = ['heart_rate', 'blood_oxygen', 'blood_pressure_systolic', 
                  'body_temperature', 'glucose_level', 'respiratory_rate']
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, vital in enumerate(key_vitals):
        if vital in df.columns:
            axes[idx].plot(df['timestamp'], df[vital], linewidth=0.8, alpha=0.7, color='steelblue')
            axes[idx].set_title(f'{vital.replace("_", " ").title()} Over Time', 
                              fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Timestamp')
            axes[idx].set_ylabel(vital.replace("_", " ").title())
            axes[idx].grid(alpha=0.3)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add normal range if available
            if vital in config.NORMAL_RANGES:
                low, high = config.NORMAL_RANGES[vital]
                axes[idx].axhline(low, color='green', linestyle='--', alpha=0.5, label='Normal Range')
                axes[idx].axhline(high, color='green', linestyle='--', alpha=0.5)
                axes[idx].fill_between(df['timestamp'], low, high, alpha=0.1, color='green')
                axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(config.EDA_DIR / 'timeseries_vitals.png', dpi=config.DPI, bbox_inches='tight')
    print(f"   ✓ Saved: timeseries_vitals.png")
    plt.close()
    
    # 4. Box plots to identify outliers
    print("\n[5/8] Generating box plots for outlier detection...")
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    axes = axes.flatten()
    
    for idx, col in enumerate(feature_cols):
        if idx < len(axes):
            bp = axes[idx].boxplot(df[col], vert=True, patch_artist=True,
                                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                                  medianprops=dict(color='red', linewidth=2),
                                  whiskerprops=dict(color='black', linewidth=1.5),
                                  capprops=dict(color='black', linewidth=1.5))
            axes[idx].set_title(f'{col.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].set_ylabel('Value')
            axes[idx].grid(alpha=0.3, axis='y')
            
            # Calculate outlier statistics
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            outlier_pct = (len(outliers) / len(df)) * 100
            axes[idx].text(0.5, 0.95, f'Outliers: {outlier_pct:.1f}%', 
                          transform=axes[idx].transAxes, ha='center', va='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for idx in range(len(feature_cols), len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(config.EDA_DIR / 'outlier_boxplots.png', dpi=config.DPI, bbox_inches='tight')
    print(f"   ✓ Saved: outlier_boxplots.png")
    plt.close()
    
    # 5. Pairwise scatter plots for highly correlated features
    print("\n[6/8] Generating pairwise scatter plots...")
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            if abs(correlation_matrix.iloc[i, j]) > 0.5:
                high_corr_pairs.append((correlation_matrix.columns[i], 
                                       correlation_matrix.columns[j],
                                       correlation_matrix.iloc[i, j]))
    
    if high_corr_pairs:
        n_pairs = min(6, len(high_corr_pairs))
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for idx, (feat1, feat2, corr) in enumerate(high_corr_pairs[:n_pairs]):
            axes[idx].scatter(df[feat1], df[feat2], alpha=0.5, s=10, color='steelblue')
            axes[idx].set_xlabel(feat1.replace("_", " ").title())
            axes[idx].set_ylabel(feat2.replace("_", " ").title())
            axes[idx].set_title(f'{feat1} vs {feat2}\nCorrelation: {corr:.3f}', 
                              fontsize=11, fontweight='bold')
            axes[idx].grid(alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_pairs, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(config.EDA_DIR / 'pairwise_scatter.png', dpi=config.DPI, bbox_inches='tight')
        print(f"   ✓ Saved: pairwise_scatter.png")
        plt.close()
    
    # 6. Health event distribution (if available)
    if 'health_event' in df.columns:
        print("\n[7/8] Analyzing health event distribution...")
        plt.figure(figsize=(10, 6))
        
        event_counts = df['health_event'].value_counts().sort_index()
        colors = ['green' if x == 0 else 'red' for x in event_counts.index]
        
        bars = plt.bar(event_counts.index, event_counts.values, color=colors, alpha=0.7, edgecolor='black')
        plt.xlabel('Health Event Type', fontsize=12, fontweight='bold')
        plt.ylabel('Count', fontsize=12, fontweight='bold')
        plt.title('Health Event Distribution', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, axis='y')
        
        # Add percentage labels
        total = len(df)
        for bar, count in zip(bars, event_counts.values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(config.EDA_DIR / 'health_event_distribution.png', dpi=config.DPI, bbox_inches='tight')
        print(f"   ✓ Saved: health_event_distribution.png")
        plt.close()
    
    # 7. Generate statistical summary
    print("\n[8/8] Generating statistical summary report...")
    
    summary_stats = df[feature_cols].describe()
    
    # Save to text file
    with open(config.EDA_DIR / 'statistical_summary.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("STATISTICAL SUMMARY - PATIENT VITAL SIGNS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Dataset Shape: {df.shape}\n")
        f.write(f"Number of Features: {len(feature_cols)}\n")
        f.write(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("DESCRIPTIVE STATISTICS\n")
        f.write("=" * 80 + "\n\n")
        f.write(summary_stats.to_string())
        f.write("\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("CORRELATION ANALYSIS\n")
        f.write("=" * 80 + "\n\n")
        f.write("Highly Correlated Feature Pairs (|r| > 0.5):\n\n")
        for feat1, feat2, corr in high_corr_pairs:
            f.write(f"  • {feat1} <-> {feat2}: {corr:.3f}\n")
        
        if 'health_event' in df.columns:
            f.write("\n" + "=" * 80 + "\n")
            f.write("HEALTH EVENT DISTRIBUTION\n")
            f.write("=" * 80 + "\n\n")
            f.write(df['health_event'].value_counts().to_string())
            f.write(f"\n\nAnomaly Rate: {(df['health_event'] != 0).sum() / len(df) * 100:.2f}%\n")
    
    print(f"   ✓ Saved: statistical_summary.txt")
    
    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)
    print(f"\nAll visualizations saved to: {config.EDA_DIR}")
    print("\nGenerated files:")
    print("  • feature_distributions.png")
    print("  • correlation_heatmap.png")
    print("  • timeseries_vitals.png")
    print("  • outlier_boxplots.png")
    print("  • pairwise_scatter.png")
    if 'health_event' in df.columns:
        print("  • health_event_distribution.png")
    print("  • statistical_summary.txt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # Run EDA
    generate_eda_report()
    print("\n✅ EDA generation successful!")
