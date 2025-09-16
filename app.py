"""Main application entry point - Production-ready Crypto Trading Dashboard."""
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import threading
import time

from dashboard.core import DashboardCore
from dashboard.charts import ChartBuilder
from config import settings
from utils.logger import get_logger, setup_logging
from models import SignalType, TrendDirection
import plotly.graph_objects as go

# Set up logging
logger = setup_logging()
logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

# Initialize core components
dashboard_core = DashboardCore()
chart_builder = ChartBuilder()

# Initialize Dash app
app = dash.Dash(__name__)
app.title = settings.APP_NAME

# Global update thread
update_thread = None
should_update = True


def _create_loading_chart(symbol: str) -> dcc.Graph:
    """Create a loading state chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Loading data for {symbol}...<br>🔄 Fetching real-time data",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="white")
    )
    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1E1E1E',
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return dcc.Graph(figure=fig, style={'marginBottom': '20px'})


def _create_error_chart(symbol: str) -> dcc.Graph:
    """Create an error state chart."""
    fig = go.Figure()
    fig.add_annotation(
        text=f"Chart error for {symbol}<br>❌ Unable to load data",
        xref="paper", yref="paper",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color="red")
    )
    fig.update_layout(
        paper_bgcolor='#0E1117',
        plot_bgcolor='#1E1E1E',
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return dcc.Graph(figure=fig, style={'marginBottom': '20px'})


def background_updater():
    """Background thread for updating data."""
    global should_update

    while should_update:
        try:
            logger.debug("Background update started")
            dashboard_core.update_dashboard_data()
            logger.debug("Background update completed")
        except Exception as e:
            logger.error(f"Error in background update: {e}")

        time.sleep(settings.UPDATE_INTERVAL / 1000)  # Convert to seconds


# Layout
app.layout = html.Div(
    style={
        'backgroundColor': '#0E1117',
        'color': '#FFFFFF',
        'fontFamily': 'Arial, sans-serif',
        'padding': '10px',
        'minHeight': '100vh'
    },
    children=[
        # Header
        html.H1(
            f"{settings.APP_NAME} - Production Version",
            style={
                'textAlign': 'center',
                'color': '#00D4FF',
                'marginBottom': '10px'
            }
        ),

        html.P(
            f"PROFESSIONAL TREND ANALYSIS • {', '.join(settings.SYMBOLS)} • "
            f"Support/Resistance • Breakout Detection • ML Predictions",
            style={
                'textAlign': 'center',
                'color': '#00FF00',
                'marginBottom': '20px',
                'fontWeight': 'bold'
            }
        ),

        # Auto-refresh
        dcc.Interval(
            id="interval",
            interval=settings.UPDATE_INTERVAL,
            n_intervals=0
        ),

        # Live Analysis Panel
        html.Div([
            html.H3("🔍 LIVE TREND ANALYSIS", style={'color': '#FF4444', 'marginBottom': '15px'}),
            html.Div(
                id="trend_analysis",
                style={
                    'fontSize': '14px',
                    'color': '#00FF00',
                    'height': '200px',
                    'overflowY': 'auto',
                    'backgroundColor': '#0A0A0A',
                    'border': '2px solid #FF4444',
                    'borderRadius': '10px',
                    'padding': '15px',
                    'fontFamily': 'monospace',
                    'marginBottom': '20px'
                }
            )
        ]),

        # Live Output Panel
        html.Div([
            html.H3("📊 LIVE TRADING OUTPUT", style={'color': '#FF4444', 'marginBottom': '15px'}),
            html.Div(
                id="live_output",
                style={
                    'fontSize': '14px',
                    'color': '#00FF00',
                    'height': '200px',
                    'overflowY': 'auto',
                    'backgroundColor': '#0A0A0A',
                    'border': '2px solid #FF4444',
                    'borderRadius': '10px',
                    'padding': '15px',
                    'fontFamily': 'monospace',
                    'marginBottom': '20px'
                }
            )
        ]),

        # Price Ticker
        html.Div([
            html.H3("💰 LIVE MARKET DATA", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(
                id="price_ticker",
                style={
                    'fontSize': '16px',
                    'color': '#FFFFFF',
                    'backgroundColor': '#1A1A1A',
                    'border': '2px solid #00D4FF',
                    'borderRadius': '10px',
                    'padding': '15px',
                    'marginBottom': '20px'
                }
            )
        ]),

        # Charts
        html.Div(id="charts", style={'marginBottom': '30px'}),

        # AI Commentary
        html.Div([
            html.H3("🤖 AI Market Commentary", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(
                id="ai_commentary",
                style={
                    'fontSize': '16px',
                    'color': '#00FF88',
                    'backgroundColor': '#1E1E1E',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'border': '2px solid #00FF88',
                    'marginBottom': '20px'
                }
            )
        ]),

        # Strategy Recommendations
        html.Div([
            html.H3("⚡ AI Strategy Recommendations", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(
                id="ai_strategy",
                style={
                    'fontSize': '18px',
                    'color': '#FFA500',
                    'backgroundColor': '#1E1E1E',
                    'padding': '15px',
                    'borderRadius': '10px',
                    'border': '2px solid #FFA500',
                    'marginBottom': '20px',
                    'fontWeight': 'bold'
                }
            )
        ]),

        # Trading Journal
        html.Div([
            html.H3("📝 Live Trading Journal", style={'color': '#00D4FF', 'marginBottom': '15px'}),
            html.Div(
                id="ai_journal",
                style={
                    'fontSize': '14px',
                    'color': '#CCCCCC',
                    'height': '350px',
                    'overflowY': 'auto',
                    'backgroundColor': '#1E1E1E',
                    'border': '2px solid #00D4FF',
                    'borderRadius': '10px',
                    'padding': '15px'
                }
            )
        ]),

        # Status Footer
        html.Div(
            id="status",
            style={
                'textAlign': 'center',
                'marginTop': '30px',
                'color': '#666666',
                'fontSize': '12px',
                'padding': '10px'
            }
        )
    ]
)


@app.callback(
    [
        Output("trend_analysis", "children"),
        Output("live_output", "children"),
        Output("price_ticker", "children"),
        Output("charts", "children"),
        Output("ai_commentary", "children"),
        Output("ai_strategy", "children"),
        Output("ai_journal", "children"),
        Output("status", "children")
    ],
    [Input("interval", "n_intervals")]
)
def update_dashboard(n):
    """Main dashboard update callback."""
    try:
        # Get current state
        state = dashboard_core.get_dashboard_state()

        # Build trend analysis display
        trend_items = []
        for symbol in settings.SYMBOLS:
            if symbol in state.trend_analyses:
                analysis = state.trend_analyses[symbol]
                summary = (
                    f"{symbol}: {analysis.current_trend.value} | "
                    f"Support: {len(analysis.support_lines)} | "
                    f"Resistance: {len(analysis.resistance_lines)} | "
                    f"Breakouts: {len(analysis.breakouts)}"
                )
                trend_items.append(
                    html.Div(summary, style={'marginBottom': '8px', 'color': '#00FFFF'})
                )

        # Build live output display
        live_messages = dashboard_core.get_live_output(25)
        live_items = []
        for msg in live_messages:
            color = '#FF0000' if '🚨' in msg else '#00FF88' if 'BUY' in msg else '#FF4444' if 'SELL' in msg else '#FFFF00'
            live_items.append(
                html.Div(msg, style={'marginBottom': '5px', 'color': color})
            )

        # Build price ticker
        ticker_items = [
            html.Div(
                f"🕐 Last Update: {state.last_update.strftime('%H:%M:%S')}",
                style={'marginBottom': '10px', 'fontWeight': 'bold'}
            )
        ]

        for symbol in settings.SYMBOLS:
            if symbol in state.market_data:
                data = state.market_data[symbol]
                change_color = '#00FF88' if data.price_change >= 0 else '#FF4444'

                # Get trend emoji
                trend_emoji = '❓'
                if symbol in state.trend_analyses:
                    trend_map = {
                        TrendDirection.STRONG_UPTREND: '🚀',
                        TrendDirection.UPTREND: '📈',
                        TrendDirection.SIDEWAYS: '➡️',
                        TrendDirection.DOWNTREND: '⬇️',
                        TrendDirection.STRONG_DOWNTREND: '📉'
                    }
                    trend_emoji = trend_map.get(
                        state.trend_analyses[symbol].current_trend, '❓'
                    )

                ticker_entry = html.Div([
                    html.Span(f"{symbol}: ", style={'fontWeight': 'bold', 'color': '#00D4FF'}),
                    html.Span(f"${data.price:.2f}", style={'fontWeight': 'bold', 'fontSize': '18px'}),
                    html.Span(f" ({data.price_change:+.2f})", style={'color': change_color, 'marginRight': '10px'}),
                    html.Span(f" {trend_emoji}", style={'fontSize': '16px'}),
                    html.Span(
                        f" ML: ${data.ml_prediction:.2f}" if data.ml_prediction else "",
                        style={'color': '#FFA500', 'marginLeft': '10px'}
                    ),
                    html.Span(
                        f" [{state.ml_status.get(symbol, 'Unknown').value}]",
                        style={'color': '#888888', 'fontSize': '12px'}
                    )
                ], style={'marginBottom': '8px'})

                ticker_items.append(ticker_entry)

        # Build optimized charts with cached data
        charts = []
        for symbol in settings.SYMBOLS:
            try:
                # Use cached data from state if available, otherwise fetch fresh
                if symbol in state.market_data:
                    # Get fresh data every 10th update to reduce API calls
                    df = dashboard_core.fetch_and_process_data(symbol)
                    if df is not None and len(df) > 0:
                        trend_analysis = state.trend_analyses.get(symbol)
                        fig = chart_builder.create_main_chart(df, symbol, trend_analysis)
                        charts.append(dcc.Graph(
                            figure=fig,
                            style={'marginBottom': '20px'},
                            config={'displayModeBar': False}  # Cleaner look
                        ))
                    else:
                        # Loading state
                        charts.append(_create_loading_chart(symbol))
                else:
                    # No data state
                    charts.append(_create_loading_chart(symbol))

            except Exception as e:
                logger.error(f"Error creating chart for {symbol}: {e}")
                charts.append(_create_error_chart(symbol))

        # Build commentary
        commentary_items = []
        for symbol in settings.SYMBOLS:
            if symbol in state.trend_analyses and symbol in state.market_data:
                analysis = state.trend_analyses[symbol]
                data = state.market_data[symbol]

                commentary = (
                    f"{symbol}: {analysis.current_trend.value} trend | "
                    f"Price: ${data.price:.2f} | "
                    f"Strength: {analysis.strength_score:.1f} | "
                    f"Lines: {len(analysis.support_lines)}S/{len(analysis.resistance_lines)}R"
                )

                if analysis.breakouts:
                    commentary += f" | 🚨 {len(analysis.breakouts)} BREAKOUTS!"

                commentary_items.append(
                    html.Div(commentary, style={'marginBottom': '10px'})
                )

        # Build strategy recommendations
        strategy_items = []
        for symbol in settings.SYMBOLS:
            if symbol in state.trading_signals and state.trading_signals[symbol]:
                latest_signal = state.trading_signals[symbol][-1]
                strategy = (
                    f"{symbol}: {latest_signal.signal.value} | "
                    f"Price: ${latest_signal.price:.2f} | "
                    f"Confidence: {latest_signal.confidence:.0f}%"
                )
                strategy_items.append(
                    html.Div(strategy, style={'marginBottom': '10px'})
                )

        # Build journal
        journal = dashboard_core.get_trading_journal()
        journal_items = []
        for symbol in settings.SYMBOLS:
            if symbol in journal and journal[symbol]:
                journal_items.append(
                    html.H4(symbol, style={'color': '#00D4FF', 'marginTop': '20px'})
                )
                for entry in journal[symbol][-5:]:  # Last 5 entries
                    color = '#00FF88' if 'BUY' in entry else '#FF4444' if 'SELL' in entry else '#CCCCCC'
                    journal_items.append(
                        html.Div(entry, style={'marginBottom': '8px', 'color': color})
                    )

        # Status
        total_symbols = len(settings.SYMBOLS)
        active_trends = len([s for s in state.trend_analyses.values() if s.current_trend != TrendDirection.INSUFFICIENT_DATA])
        total_breakouts = sum(len(a.breakouts) for a in state.trend_analyses.values())

        status = (
            f"⚡ Last update: {state.last_update.strftime('%H:%M:%S')} | "
            f"Symbols: {total_symbols} | Active trends: {active_trends} | "
            f"Breakouts: {total_breakouts} | Version: {settings.APP_VERSION}"
        )

        return (
            trend_items or [html.Div("Analyzing trends...", style={'fontStyle': 'italic'})],
            live_items or [html.Div("Waiting for live data...", style={'fontStyle': 'italic'})],
            ticker_items,
            charts,
            commentary_items or [html.Div("Generating commentary...", style={'fontStyle': 'italic'})],
            strategy_items or [html.Div("Analyzing strategies...", style={'fontStyle': 'italic'})],
            journal_items or [html.Div("No trading signals yet...", style={'fontStyle': 'italic'})],
            status
        )

    except Exception as e:
        logger.error(f"Dashboard update error: {e}")
        error_msg = f"Error: {str(e)[:100]}..."
        return [html.Div(error_msg)] * 8


if __name__ == "__main__":
    # Start background update thread
    update_thread = threading.Thread(target=background_updater, daemon=True)
    update_thread.start()

    logger.info("=" * 80)
    logger.info(f"🚀 {settings.APP_NAME.upper()} - PRODUCTION VERSION")
    logger.info("=" * 80)
    logger.info(f"📊 Symbols: {', '.join(settings.SYMBOLS)}")
    logger.info(f"⏱️  Update Interval: {settings.UPDATE_INTERVAL/1000} seconds")
    logger.info(f"🌐 Dashboard URL: http://{settings.HOST}:{settings.PORT}")
    logger.info("=" * 80)

    try:
        app.run(
            debug=settings.DEBUG,
            host=settings.HOST,
            port=settings.PORT
        )
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        should_update = False
    finally:
        should_update = False
        if update_thread and update_thread.is_alive():
            update_thread.join(timeout=1)
        logger.info("Dashboard stopped")