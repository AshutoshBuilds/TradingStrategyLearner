import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Visualizer:
    def __init__(self, style: str = 'darkgrid', figsize: tuple = (12, 8)):
        """Initialize the visualizer.
        
        Args:
            style (str): Seaborn style theme
            figsize (tuple): Default figure size
        """
        self.style = style
        self.figsize = figsize
        self._setup_style()

    def _setup_style(self):
        """Set up visualization style and defaults."""
        sns.set_style(self.style)
        plt.rcParams['figure.figsize'] = self.figsize
        
        # Custom color palette
        self.colors = {
            'primary': '#2962FF',
            'secondary': '#FF6D00',
            'positive': '#00C853',
            'negative': '#D50000',
            'neutral': '#757575'
        }

    def plot_price_analysis(self, data: Dict[str, Any], title: str = "Price Analysis") -> None:
        """Plot price analysis with technical indicators.
        
        Args:
            data (Dict[str, Any]): Price and indicator data
            title (str): Plot title
        """
        fig = make_subplots(rows=2, cols=1, shared_xaxis=True,
                           vertical_spacing=0.05,
                           subplot_titles=(title, "Volume"))
        
        # Price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data['dates'],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="Price"
            ),
            row=1, col=1
        )
        
        # Add technical indicators if available
        if 'sma' in data:
            fig.add_trace(
                go.Scatter(
                    x=data['dates'],
                    y=data['sma'],
                    name="SMA",
                    line=dict(color='orange')
                ),
                row=1, col=1
            )
            
        if 'ema' in data:
            fig.add_trace(
                go.Scatter(
                    x=data['dates'],
                    y=data['ema'],
                    name="EMA",
                    line=dict(color='blue')
                ),
                row=1, col=1
            )
            
        # Volume bars
        colors = ['red' if close < open else 'green'
                 for close, open in zip(data['close'], data['open'])]
        
        fig.add_trace(
            go.Bar(
                x=data['dates'],
                y=data['volume'],
                name="Volume",
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            title_text=title
        )
        
        fig.show()

    def plot_sentiment_analysis(self, sentiments: List[Dict[str, Any]],
                              title: str = "Sentiment Analysis") -> None:
        """Plot sentiment analysis results.
        
        Args:
            sentiments (List[Dict[str, Any]]): Sentiment analysis data
            title (str): Plot title
        """
        # Extract data
        labels = [s['label'] for s in sentiments]
        scores = [s['score'] for s in sentiments]
        dates = [s.get('date', i) for i, s in enumerate(sentiments)]
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Plot sentiment scores
        colors = [self.colors['positive'] if label == 'POSITIVE'
                 else self.colors['negative'] for label in labels]
        
        plt.bar(dates, scores, color=colors)
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Sentiment Score')
        plt.ylim(0, 1)
        
        # Add legend
        legend_elements = [plt.Rectangle((0,0),1,1, color=self.colors['positive'], label='Positive'),
                         plt.Rectangle((0,0),1,1, color=self.colors['negative'], label='Negative')]
        plt.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.show()

    def plot_topic_distribution(self, topics: List[Dict[str, Any]],
                              title: str = "Topic Distribution") -> None:
        """Plot distribution of extracted topics.
        
        Args:
            topics (List[Dict[str, Any]]): Topic data
            title (str): Plot title
        """
        # Extract data
        terms = [t['main_term'] for t in topics]
        frequencies = [t['frequency'] for t in topics]
        
        # Sort by frequency
        sorted_indices = np.argsort(frequencies)[::-1]
        terms = [terms[i] for i in sorted_indices]
        frequencies = [frequencies[i] for i in sorted_indices]
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Plot horizontal bars
        y_pos = np.arange(len(terms))
        plt.barh(y_pos, frequencies, color=self.colors['primary'])
        
        # Customize plot
        plt.yticks(y_pos, terms)
        plt.xlabel('Frequency')
        plt.title(title)
        
        plt.tight_layout()
        plt.show()

    def plot_strategy_evaluation(self, evaluation: Dict[str, Any],
                               title: str = "Strategy Evaluation") -> None:
        """Plot strategy evaluation metrics.
        
        Args:
            evaluation (Dict[str, Any]): Strategy evaluation metrics
            title (str): Plot title
        """
        # Extract metrics
        metrics = list(evaluation.keys())
        values = list(evaluation.values())
        
        # Create figure
        fig = go.Figure()
        
        # Add radar plot
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics,
            fill='toself',
            name='Strategy Metrics'
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(values)]
                )
            ),
            showlegend=False,
            title=title
        )
        
        fig.show()

    def plot_market_conditions(self, conditions: Dict[str, Any],
                             title: str = "Market Conditions") -> None:
        """Plot current market conditions.
        
        Args:
            conditions (Dict[str, Any]): Market condition data
            title (str): Plot title
        """
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # Plot trend and volatility
        trend_data = pd.Series(conditions['trend_data'])
        volatility_data = pd.Series(conditions['volatility_data'])
        
        trend_data.plot(ax=ax1, color=self.colors['primary'], label='Price Trend')
        ax1.set_title('Price Trend')
        ax1.legend()
        
        volatility_data.plot(ax=ax2, color=self.colors['secondary'], label='Volatility')
        ax2.set_title('Volatility')
        ax2.legend()
        
        # Add market regime annotation
        regime = conditions.get('market_regime', 'Unknown')
        fig.text(0.02, 0.98, f'Market Regime: {regime}', fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def plot_strategy_conditions(self, conditions: List[Dict[str, Any]],
                               market_data: Dict[str, Any],
                               title: str = "Strategy Conditions") -> None:
        """Plot strategy conditions against market data.
        
        Args:
            conditions (List[Dict[str, Any]]): Strategy conditions
            market_data (Dict[str, Any]): Market data
            title (str): Plot title
        """
        # Create figure with subplots
        fig = make_subplots(rows=len(conditions), cols=1,
                           subplot_titles=[c['type'] for c in conditions])
        
        # Plot each condition
        for i, condition in enumerate(conditions, 1):
            if condition['type'] in ['trend', 'deviation', 'range']:
                # Plot price data
                fig.add_trace(
                    go.Scatter(
                        x=market_data['dates'],
                        y=market_data[condition['indicator']],
                        name=condition['indicator'].capitalize(),
                        line=dict(color=self.colors['primary'])
                    ),
                    row=i, col=1
                )
                
                # Plot threshold lines
                if 'threshold' in condition:
                    threshold = condition['threshold']
                    fig.add_hline(
                        y=threshold,
                        line_dash="dash",
                        line_color=self.colors['secondary'],
                        row=i, col=1
                    )
                    
            elif condition['type'] in ['sentiment', 'topic']:
                # Plot sentiment/topic scores
                fig.add_trace(
                    go.Bar(
                        x=[condition['value']] if 'value' in condition else condition['values'],
                        y=[condition['confidence']],
                        name=condition['type'].capitalize(),
                        marker_color=self.colors['primary']
                    ),
                    row=i, col=1
                )
                
        # Update layout
        fig.update_layout(
            height=300 * len(conditions),
            showlegend=True,
            title_text=title
        )
        
        fig.show()

    def plot_backtest_results(self, results: Dict[str, Any],
                            title: str = "Backtest Results") -> None:
        """Plot backtest results.
        
        Args:
            results (Dict[str, Any]): Backtest results data
            title (str): Plot title
        """
        # Create figure with subplots
        fig = make_subplots(rows=2, cols=1,
                           subplot_titles=("Equity Curve", "Drawdown"))
        
        # Plot equity curve
        fig.add_trace(
            go.Scatter(
                x=results['dates'],
                y=results['equity_curve'],
                name="Equity",
                line=dict(color=self.colors['primary'])
            ),
            row=1, col=1
        )
        
        # Plot drawdown
        fig.add_trace(
            go.Scatter(
                x=results['dates'],
                y=results['drawdown'],
                name="Drawdown",
                fill='tozeroy',
                line=dict(color=self.colors['negative'])
            ),
            row=2, col=1
        )
        
        # Add performance metrics
        metrics_text = (
            f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}<br>"
            f"Win Rate: {results['metrics']['win_rate']:.2%}<br>"
            f"Max Drawdown: {results['metrics']['max_drawdown']:.2%}"
        )
        
        fig.add_annotation(
            text=metrics_text,
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            showarrow=False,
            font=dict(size=10)
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text=title
        )
        
        fig.show()

    def plot_trade_distribution(self, trades: List[Dict[str, Any]],
                              title: str = "Trade Distribution") -> None:
        """Plot distribution of trade results.
        
        Args:
            trades (List[Dict[str, Any]]): Trade data
            title (str): Plot title
        """
        # Extract trade returns
        returns = [t['return'] for t in trades]
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Plot histogram of returns
        sns.histplot(returns, bins=50, color=self.colors['primary'])
        
        # Add mean and std lines
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        plt.axvline(mean_return, color=self.colors['secondary'], linestyle='--',
                   label=f'Mean: {mean_return:.2%}')
        plt.axvline(mean_return + std_return, color=self.colors['neutral'], linestyle=':',
                   label=f'+1 Std: {(mean_return + std_return):.2%}')
        plt.axvline(mean_return - std_return, color=self.colors['neutral'], linestyle=':',
                   label=f'-1 Std: {(mean_return - std_return):.2%}')
        
        # Customize plot
        plt.title(title)
        plt.xlabel('Return')
        plt.ylabel('Count')
        plt.legend()
        
        plt.tight_layout()
        plt.show()

    def save_plot(self, filename: str, format: str = 'png', dpi: int = 300) -> None:
        """Save the current plot to a file.
        
        Args:
            filename (str): Output filename
            format (str): Output format (png, pdf, svg)
            dpi (int): Resolution for raster formats
        """
        plt.savefig(filename, format=format, dpi=dpi, bbox_inches='tight') 