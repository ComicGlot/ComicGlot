import os
import pandas as pd
import yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
from pathlib import Path
import base64
from PIL import Image
import io

class YOLOTrainingAnalyzer:
    def __init__(self, runs_path):
        self.runs_path = Path(runs_path)
        self.training_data = {}
        
    def scan_training_runs(self):
        """Scan all training folders and extract data"""
        print(f"Scanning {self.runs_path} for training runs...")
        
        # Look for detect folder or scan directly
        detect_path = self.runs_path / "detect"
        if detect_path.exists():
            scan_path = detect_path
        else:
            scan_path = self.runs_path
            
        for folder in scan_path.iterdir():
            if folder.is_dir() and folder.name.startswith('train'):
                self._process_training_folder(folder)
                
        print(f"Found {len(self.training_data)} training runs")
        return self.training_data
    
    def _process_training_folder(self, folder_path):
        """Process individual training folder"""
        try:
            # Read args.yaml
            args_file = folder_path / "args.yaml"
            results_file = folder_path / "results.csv"
            
            if not args_file.exists() or not results_file.exists():
                print(f"Skipping {folder_path.name}: missing args.yaml or results.csv")
                return
                
            # Load configuration
            with open(args_file, 'r') as f:
                args = yaml.safe_load(f)
                
            # Load results
            results_df = pd.read_csv(results_file)
            
            # Find confusion matrix images
            conf_matrices = self._find_confusion_matrices(folder_path)
            
            # Store data
            self.training_data[folder_path.name] = {
                'args': args,
                'results': results_df,
                'folder_path': folder_path,
                'confusion_matrices': conf_matrices,
                'model': args.get('model', 'unknown'),
                'epochs': args.get('epochs', len(results_df)),
                'batch_size': args.get('batch', 'unknown'),
                'imgsz': args.get('imgsz', 'unknown'),
                'final_map50': results_df['metrics/mAP50(B)'].iloc[-1] if len(results_df) > 0 else 0,
                'final_map50_95': results_df['metrics/mAP50-95(B)'].iloc[-1] if len(results_df) > 0 else 0,
            }
            
            print(f"Processed {folder_path.name}: {args.get('model', 'unknown')} - {len(results_df)} epochs")
            
        except Exception as e:
            print(f"Error processing {folder_path.name}: {e}")
    
    def _find_confusion_matrices(self, folder_path):
        """Find confusion matrix images in the folder"""
        conf_files = {}
        for file in folder_path.glob("*confusion_matrix*"):
            if file.suffix in ['.png', '.jpg', '.jpeg']:
                conf_files[file.stem] = str(file)
        return conf_files
    
    def _image_to_base64(self, image_path):
        """Convert image to base64 for embedding in plotly"""
        try:
            with Image.open(image_path) as img:
                # Resize if too large
                if img.width > 800 or img.height > 800:
                    img.thumbnail((800, 800), Image.Resampling.LANCZOS)
                
                buffer = io.BytesIO()
                img.save(buffer, format='PNG')
                img_str = base64.b64encode(buffer.getvalue()).decode()
                return f"data:image/png;base64,{img_str}"
        except Exception as e:
            print(f"Error converting image {image_path}: {e}")
            return None
    
    def create_training_comparison_plot(self):
        """Create comprehensive training comparison plots"""
        if not self.training_data:
            print("No training data found!")
            return None
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('mAP@0.5 Progress', 'mAP@0.5:0.95 Progress', 
                          'Training Loss', 'Validation Loss'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (run_name, data) in enumerate(self.training_data.items()):
            color = colors[i % len(colors)]
            results = data['results']
            
            # Hover text with detailed info
            hover_text = [
                f"<b>{run_name}</b><br>" +
                f"Model: {data['model']}<br>" +
                f"Epoch: {epoch+1}/{data['epochs']}<br>" +
                f"Batch Size: {data['batch_size']}<br>" +
                f"Image Size: {data['imgsz']}<br>" +
                f"mAP@0.5: {map50:.4f}<br>" +
                f"mAP@0.5:0.95: {map50_95:.4f}"
                for epoch, (map50, map50_95) in enumerate(zip(
                    results['metrics/mAP50(B)'], 
                    results['metrics/mAP50-95(B)']
                ))
            ]
            
            # mAP@0.5 plot
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=results['metrics/mAP50(B)'],
                    mode='lines+markers',
                    name=f"{run_name} (mAP@0.5)",
                    line=dict(color=color),
                    hovertemplate=hover_text,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # mAP@0.5:0.95 plot
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=results['metrics/mAP50-95(B)'],
                    mode='lines+markers',
                    name=f"{run_name} (mAP@0.5:0.95)",
                    line=dict(color=color, dash='dash'),
                    hovertemplate=hover_text,
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Training Loss
            train_loss = (results['train/box_loss'] + 
                         results['train/cls_loss'] + 
                         results['train/dfl_loss'])
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=train_loss,
                    mode='lines+markers',
                    name=f"{run_name} (Train Loss)",
                    line=dict(color=color),
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Validation Loss
            val_loss = (results['val/box_loss'] + 
                       results['val/cls_loss'] + 
                       results['val/dfl_loss'])
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=val_loss,
                    mode='lines+markers',
                    name=f"{run_name} (Val Loss)",
                    line=dict(color=color, dash='dot'),
                    showlegend=False
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            title="YOLO Training Runs Comparison",
            height=800,
            hovermode='closest',
            template='plotly_white'
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        
        fig.update_yaxes(title_text="mAP@0.5", row=1, col=1)
        fig.update_yaxes(title_text="mAP@0.5:0.95", row=1, col=2)
        fig.update_yaxes(title_text="Training Loss", row=2, col=1)
        fig.update_yaxes(title_text="Validation Loss", row=2, col=2)
        
        return fig
    
    def create_final_results_comparison(self):
        """Create bar chart comparing final results"""
        if not self.training_data:
            return None
            
        runs = list(self.training_data.keys())
        map50_scores = [data['final_map50'] for data in self.training_data.values()]
        map50_95_scores = [data['final_map50_95'] for data in self.training_data.values()]
        models = [data['model'] for data in self.training_data.values()]
        epochs = [data['epochs'] for data in self.training_data.values()]
        
        fig = go.Figure()
        
        # mAP@0.5 bars
        fig.add_trace(go.Bar(
            name='mAP@0.5',
            x=runs,
            y=map50_scores,
            text=[f"{score:.3f}" for score in map50_scores],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                         "Model: %{customdata[0]}<br>" +
                         "Epochs: %{customdata[1]}<br>" +
                         "mAP@0.5: %{y:.4f}<extra></extra>",
            customdata=list(zip(models, epochs))
        ))
        
        # mAP@0.5:0.95 bars
        fig.add_trace(go.Bar(
            name='mAP@0.5:0.95',
            x=runs,
            y=map50_95_scores,
            text=[f"{score:.3f}" for score in map50_95_scores],
            textposition='auto',
            hovertemplate="<b>%{x}</b><br>" +
                         "Model: %{customdata[0]}<br>" +
                         "Epochs: %{customdata[1]}<br>" +
                         "mAP@0.5:0.95: %{y:.4f}<extra></extra>",
            customdata=list(zip(models, epochs))
        ))
        
        fig.update_layout(
            title="Final Model Performance Comparison",
            xaxis_title="Training Run",
            yaxis_title="mAP Score",
            barmode='group',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def create_detailed_metrics_plot(self):
        """Create detailed metrics comparison plot"""
        if not self.training_data:
            return None
            
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Precision', 'Recall', 'Box Loss', 'Class Loss')
        )
        
        colors = px.colors.qualitative.Set1
        
        for i, (run_name, data) in enumerate(self.training_data.items()):
            color = colors[i % len(colors)]
            results = data['results']
            
            # Precision
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=results['metrics/precision(B)'],
                    mode='lines+markers',
                    name=f"{run_name}",
                    line=dict(color=color),
                    legendgroup=run_name,
                    showlegend=True
                ),
                row=1, col=1
            )
            
            # Recall
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=results['metrics/recall(B)'],
                    mode='lines+markers',
                    name=f"{run_name}",
                    line=dict(color=color),
                    legendgroup=run_name,
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Box Loss
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=results['train/box_loss'],
                    mode='lines+markers',
                    name=f"{run_name}",
                    line=dict(color=color),
                    legendgroup=run_name,
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Class Loss
            fig.add_trace(
                go.Scatter(
                    x=results['epoch'],
                    y=results['train/cls_loss'],
                    mode='lines+markers',
                    name=f"{run_name}",
                    line=dict(color=color),
                    legendgroup=run_name,
                    showlegend=False
                ),
                row=2, col=2
            )
        
        fig.update_layout(
            title="Detailed Training Metrics Comparison",
            height=800,
            template='plotly_white'
        )
        
        return fig
    
    def generate_summary_report(self):
        """Generate a text summary of all training runs"""
        if not self.training_data:
            return "No training data found."
            
        report = "=== YOLO Training Runs Summary ===\n\n"
        
        # Sort by final mAP@0.5
        sorted_runs = sorted(
            self.training_data.items(),
            key=lambda x: x[1]['final_map50'],
            reverse=True
        )
        
        for i, (run_name, data) in enumerate(sorted_runs, 1):
            report += f"{i}. {run_name}\n"
            report += f"   Model: {data['model']}\n"
            report += f"   Epochs: {data['epochs']}\n"
            report += f"   Batch Size: {data['batch_size']}\n"
            report += f"   Image Size: {data['imgsz']}\n"
            report += f"   Final mAP@0.5: {data['final_map50']:.4f}\n"
            report += f"   Final mAP@0.5:0.95: {data['final_map50_95']:.4f}\n"
            if data['confusion_matrices']:
                report += f"   Confusion Matrices: {len(data['confusion_matrices'])} found\n"
            report += "\n"
            
        return report


def main():
    """Main function to run the analyzer"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze YOLO training runs')
    parser.add_argument('runs_path', help='Path to runs folder')
    parser.add_argument('--output', '-o', default='yolo_analysis.html', 
                       help='Output HTML file name')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = YOLOTrainingAnalyzer(args.runs_path)
    
    # Scan for training runs
    analyzer.scan_training_runs()
    
    if not analyzer.training_data:
        print("No training runs found!")
        return
    
    # Create plots
    print("Creating comparison plots...")
    
    fig1 = analyzer.create_training_comparison_plot()
    fig2 = analyzer.create_final_results_comparison()
    fig3 = analyzer.create_detailed_metrics_plot()
    
    # Combine all plots into one HTML file
    from plotly.offline import plot
    import plotly.graph_objects as go
    
    # Create a combined HTML with all plots
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>YOLO Training Analysis</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .plot-container {{ margin: 30px 0; }}
            .summary {{ background-color: #f5f5f5; padding: 15px; margin: 20px 0; white-space: pre-line; }}
        </style>
    </head>
    <body>
        <h1>YOLO Training Runs Analysis</h1>
        
        <div class="summary">
            <h2>Summary Report</h2>
            {analyzer.generate_summary_report()}
        </div>
        
        <div class="plot-container" id="plot1"></div>
        <div class="plot-container" id="plot2"></div>
        <div class="plot-container" id="plot3"></div>
        
        <script>
            Plotly.newPlot('plot1', {fig1.to_json()});
            Plotly.newPlot('plot2', {fig2.to_json()});
            Plotly.newPlot('plot3', {fig3.to_json()});
        </script>
    </body>
    </html>
    """
    
    # Save individual plots
    if fig1:
        fig1.write_html(f"training_comparison_{args.output}")
        fig1.show()
    
    if fig2:
        fig2.write_html(f"final_results_{args.output}")
        fig2.show()
        
    if fig3:
        fig3.write_html(f"detailed_metrics_{args.output}")
        fig3.show()
    
    # Print summary
    print("\n" + analyzer.generate_summary_report())
    
    print(f"\nAnalysis complete! Plots saved and displayed.")


if __name__ == "__main__":
    main()