"""Chart building utilities for the dashboard."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional

from models import TrendAnalysis, BreakoutSignal
from utils.logger import get_logger

logger = get_logger("dashboard_charts")


class ChartBuilder:
    """Utility class for building trading charts."""

    def __init__(self):
        self.default_colors = {
            'candlestick_up': '#00FF88',
            'candlestick_down': '#FF4444',
            'support_strong': '#00FF00',
            'support_medium': '#66FF66',
            'support_weak': '#99FF99',
            'resistance_strong': '#FF0000',
            'resistance_medium': '#FF6666',
            'resistance_weak': '#FF9999',
            'ma_20': 'orange',
            'ma_50': 'yellow',
            'bb_bands': 'rgba(0,150,255,0.6)',
            'rsi': 'purple',
            'macd': 'blue',
            'macd_signal': 'red'
        }

    def create_main_chart(
        self,
        df: pd.DataFrame,
        symbol: str,
        trend_analysis: Optional[TrendAnalysis] = None
    ) -> go.Figure:
        """Create the main trading chart with trend lines."""
        try:
            # Create subplots
            fig = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    f'{symbol} - Price, Trend Lines & Indicators',
                    'RSI (14)',
                    'MACD'
                ),
                row_heights=[0.6, 0.2, 0.2]
            )

            # Main candlestick chart
            fig.add_trace(go.Candlestick(
                x=df['timestamp'],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price',
                increasing_line_color=self.default_colors['candlestick_up'],
                decreasing_line_color=self.default_colors['candlestick_down']
            ), row=1, col=1)

            # Add trend lines if available
            if trend_analysis:
                self._add_trend_lines(fig, df, trend_analysis, row=1, col=1)
                self._add_breakout_annotations(fig, df, trend_analysis, row=1, col=1)

            # Add technical indicators
            self._add_moving_averages(fig, df, row=1, col=1)
            self._add_bollinger_bands(fig, df, row=1, col=1)
            self._add_trading_signals(fig, df, row=1, col=1)

            # RSI subplot
            if 'RSI' in df.columns:
                self._add_rsi(fig, df, row=2, col=1)

            # MACD subplot
            if 'MACD' in df.columns:
                self._add_macd(fig, df, row=3, col=1)

            # Update layout
            self._update_chart_layout(fig, symbol)

            return fig

        except Exception as e:
            logger.error(f"Error creating chart for {symbol}: {e}")
            # Return empty chart on error
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error loading chart for {symbol}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig

    def _add_trend_lines(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        trend_analysis: TrendAnalysis,
        row: int = 1,
        col: int = 1
    ) -> None:
        """Add trend lines to the chart."""
        try:
            # Add resistance lines
            for i, line in enumerate(trend_analysis.resistance_lines):
                start_idx = max(0, line.start_idx)
                end_idx = min(len(df) - 1, line.end_idx)

                x_values = [df.iloc[start_idx]['timestamp'], df.iloc[end_idx]['timestamp']]
                y_values = [
                    line.slope * start_idx + line.intercept,
                    line.slope * end_idx + line.intercept
                ]

                # Color and width based on strength
                color, width = self._get_line_style(line.strength, 'resistance')

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name=f'Resistance {i+1} (S:{line.strength:.0f})',
                    line=dict(color=color, width=width, dash='solid'),
                    hovertemplate=(
                        f'<b>Resistance Line {i+1}</b><br>'
                        f'Strength: {line.strength:.1f}<br>'
                        f'Touches: {line.touches}<br>'
                        f'R²: {line.r_squared:.3f}<br>'
                        f'Price: $%{{y:.2f}}<extra></extra>'
                    ),
                    showlegend=True
                ), row=row, col=col)

            # Add support lines
            for i, line in enumerate(trend_analysis.support_lines):
                start_idx = max(0, line.start_idx)
                end_idx = min(len(df) - 1, line.end_idx)

                x_values = [df.iloc[start_idx]['timestamp'], df.iloc[end_idx]['timestamp']]
                y_values = [
                    line.slope * start_idx + line.intercept,
                    line.slope * end_idx + line.intercept
                ]

                color, width = self._get_line_style(line.strength, 'support')

                fig.add_trace(go.Scatter(
                    x=x_values,
                    y=y_values,
                    mode='lines',
                    name=f'Support {i+1} (S:{line.strength:.0f})',
                    line=dict(color=color, width=width, dash='solid'),
                    hovertemplate=(
                        f'<b>Support Line {i+1}</b><br>'
                        f'Strength: {line.strength:.1f}<br>'
                        f'Touches: {line.touches}<br>'
                        f'R²: {line.r_squared:.3f}<br>'
                        f'Price: $%{{y:.2f}}<extra></extra>'
                    ),
                    showlegend=True
                ), row=row, col=col)

        except Exception as e:
            logger.error(f"Error adding trend lines: {e}")

    def _add_breakout_annotations(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        trend_analysis: TrendAnalysis,
        row: int = 1,
        col: int = 1
    ) -> None:
        """Add breakout annotations to the chart."""
        try:
            if not trend_analysis.breakouts or df.empty:
                return

            latest_timestamp = df.iloc[-1]['timestamp']
            latest_price = df.iloc[-1]['close']

            for breakout in trend_analysis.breakouts:
                if breakout.type.value == 'resistance_breakout':
                    fig.add_annotation(
                        x=latest_timestamp,
                        y=latest_price * 1.005,
                        text=f"🚀 BREAKOUT!<br>Confidence: {breakout.confidence:.0f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#00FF00",
                        bgcolor="#00FF00",
                        bordercolor="#00FF00",
                        font=dict(color="black", size=12, family="Arial Black"),
                        borderwidth=2,
                        row=row, col=col
                    )
                elif breakout.type.value == 'support_breakdown':
                    fig.add_annotation(
                        x=latest_timestamp,
                        y=latest_price * 0.995,
                        text=f"💥 BREAKDOWN!<br>Confidence: {breakout.confidence:.0f}%",
                        showarrow=True,
                        arrowhead=2,
                        arrowcolor="#FF0000",
                        bgcolor="#FF0000",
                        bordercolor="#FF0000",
                        font=dict(color="white", size=12, family="Arial Black"),
                        borderwidth=2,
                        row=row, col=col
                    )

        except Exception as e:
            logger.error(f"Error adding breakout annotations: {e}")

    def _add_moving_averages(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        row: int = 1,
        col: int = 1
    ) -> None:
        """Add moving averages to the chart."""
        try:
            if 'MA_20' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MA_20'],
                    mode='lines', name='MA20',
                    line=dict(color=self.default_colors['ma_20'], width=2)
                ), row=row, col=col)

            if 'MA_50' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['MA_50'],
                    mode='lines', name='MA50',
                    line=dict(color=self.default_colors['ma_50'], width=2)
                ), row=row, col=col)

        except Exception as e:
            logger.error(f"Error adding moving averages: {e}")

    def _add_bollinger_bands(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        row: int = 1,
        col: int = 1
    ) -> None:
        """Add Bollinger Bands to the chart."""
        try:
            if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['BB_upper'],
                    mode='lines', name='BB Upper',
                    line=dict(color=self.default_colors['bb_bands'], width=1, dash='dash')
                ), row=row, col=col)

                fig.add_trace(go.Scatter(
                    x=df['timestamp'], y=df['BB_lower'],
                    mode='lines', name='BB Lower',
                    line=dict(color=self.default_colors['bb_bands'], width=1, dash='dash'),
                    fill='tonexty', fillcolor='rgba(0,150,255,0.1)'
                ), row=row, col=col)

        except Exception as e:
            logger.error(f"Error adding Bollinger Bands: {e}")

    def _add_trading_signals(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        row: int = 1,
        col: int = 1
    ) -> None:
        """Add trading signal annotations to the chart."""
        try:
            if 'Signal' not in df.columns:
                return

            buy_signals = df[df['Signal'] == 'BUY']
            sell_signals = df[df['Signal'] == 'SELL']

            for _, signal_row in buy_signals.iterrows():
                fig.add_annotation(
                    x=signal_row['timestamp'],
                    y=signal_row['low'] * 0.998,
                    text="📈 BUY",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="green",
                    bgcolor="green",
                    bordercolor="green",
                    font=dict(color="white", size=12, family="Arial Black"),
                    row=row, col=col
                )

            for _, signal_row in sell_signals.iterrows():
                fig.add_annotation(
                    x=signal_row['timestamp'],
                    y=signal_row['high'] * 1.002,
                    text="📉 SELL",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor="red",
                    bgcolor="red",
                    bordercolor="red",
                    font=dict(color="white", size=12, family="Arial Black"),
                    row=row, col=col
                )

        except Exception as e:
            logger.error(f"Error adding trading signals: {e}")

    def _add_rsi(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        row: int = 2,
        col: int = 1
    ) -> None:
        """Add RSI to the chart."""
        try:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['RSI'],
                mode='lines', name='RSI',
                line=dict(color=self.default_colors['rsi'], width=2)
            ), row=row, col=col)

            # RSI reference lines
            fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.7, row=row, col=col)
            fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.7, row=row, col=col)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.5, row=row, col=col)

        except Exception as e:
            logger.error(f"Error adding RSI: {e}")

    def _add_macd(
        self,
        fig: go.Figure,
        df: pd.DataFrame,
        row: int = 3,
        col: int = 1
    ) -> None:
        """Add MACD to the chart."""
        try:
            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['MACD'],
                mode='lines', name='MACD',
                line=dict(color=self.default_colors['macd'], width=2)
            ), row=row, col=col)

            fig.add_trace(go.Scatter(
                x=df['timestamp'], y=df['MACD_signal'],
                mode='lines', name='Signal',
                line=dict(color=self.default_colors['macd_signal'], width=2)
            ), row=row, col=col)

            if 'MACD_histogram' in df.columns:
                fig.add_trace(go.Bar(
                    x=df['timestamp'], y=df['MACD_histogram'],
                    name='Histogram',
                    marker_color=['green' if x >= 0 else 'red' for x in df['MACD_histogram']],
                    opacity=0.6
                ), row=row, col=col)

        except Exception as e:
            logger.error(f"Error adding MACD: {e}")

    def _get_line_style(self, strength: float, line_type: str) -> tuple:
        """Get line color and width based on strength."""
        if line_type == 'resistance':
            if strength > 80:
                return self.default_colors['resistance_strong'], 3
            elif strength > 60:
                return self.default_colors['resistance_medium'], 2
            else:
                return self.default_colors['resistance_weak'], 1
        else:  # support
            if strength > 80:
                return self.default_colors['support_strong'], 3
            elif strength > 60:
                return self.default_colors['support_medium'], 2
            else:
                return self.default_colors['support_weak'], 1

    def _update_chart_layout(self, fig: go.Figure, symbol: str) -> None:
        """Update chart layout with professional styling."""
        fig.update_layout(
            height=800,
            showlegend=True,
            paper_bgcolor='#0E1117',
            plot_bgcolor='#1E1E1E',
            font=dict(color='white'),
            margin=dict(l=50, r=50, t=80, b=50),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(0,0,0,0.8)",
                bordercolor="rgba(255,255,255,0.2)",
                borderwidth=1
            ),
            title=dict(
                text=f"{symbol} Professional Trading Analysis",
                x=0.5,
                font=dict(size=16, color='#00D4FF')
            )
        )

        fig.update_xaxes(gridcolor='#333333')
        fig.update_yaxes(gridcolor='#333333')