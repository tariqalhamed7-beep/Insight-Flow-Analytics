#!/usr/bin/env python3
"""
Transaction Congestion Predictor - Streamlit Dashboard

This module provides a web dashboard for visualizing transaction congestion predictions.
It connects to the FastAPI backend and displays predictions visually.

Author: Senior Full-Stack Developer
"""

import os
import sys
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# Configuration
# =============================================================================

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_TIMEOUT = 30

# Page configuration
st.set_page_config(
    page_title="Transaction Congestion Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1.5rem;
        text-align: center;
    }
    .congestion-low { color: #28a745; }
    .congestion-medium { color: #ffc107; }
    .congestion-high { color: #fd7e14; }
    .congestion-critical { color: #dc3545; }
    .stMetric label {
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# API Client Functions
# =============================================================================

def check_api_health() -> bool:
    """Check if the API is accessible."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_prediction(date: str, hour: int = 12) -> Optional[Dict[str, Any]]:
    """Get a single prediction from the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/date",
            params={"date": date, "hour": hour},
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the FastAPI server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_batch_predictions(dates: List[str]) -> Optional[List[Dict[str, Any]]]:
    """Get batch predictions from the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json={"features_list": dates},
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.text}")
            return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None


def get_api_info() -> Optional[Dict[str, Any]]:
    """Get API information."""
    try:
        response = requests.get(f"{API_BASE_URL}/info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def build_features_from_date(date_str: str, hour: int = 12) -> Dict[str, Any]:
    """Build features dictionary from a date string."""
    date = datetime.strptime(date_str, "%Y-%m-%d")
    return {
        "day_of_week": date.weekday(),
        "day_of_month": date.day,
        "month": date.month,
        "quarter": (date.month - 1) // 3 + 1,
        "is_weekend": 1 if date.weekday() >= 5 else 0,
        "is_month_start": 1 if date.day <= 3 else 0,
        "is_month_end": 1 if date.day >= 28 else 0,
        "is_quarter_start": 1 if date.month in [1, 4, 7, 10] and date.day <= 3 else 0,
        "is_quarter_end": 1 if date.month in [3, 6, 9, 12] and date.day >= 28 else 0,
        "week_of_year": date.isocalendar()[1],
        "days_since_epoch": (date - datetime(2014, 1, 1)).days
    }


# =============================================================================
# UI Components
# =============================================================================

def render_header():
    """Render the page header."""
    st.markdown('<h1 class="main-header">📊 Transaction Congestion Predictor</h1>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar() -> tuple:
    """Render sidebar with API status and controls."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Status
        st.subheader("API Status")
        if check_api_health():
            st.success("✅ API Connected")
        else:
            st.error("❌ API Unavailable")
            st.info(f"Please ensure the FastAPI server is running at:\n`{API_BASE_URL}`")
        
        st.markdown("---")
        
        # Quick info
        st.subheader("📋 Info")
        api_info = get_api_info()
        if api_info:
            with st.expander("Congestion Thresholds", expanded=False):
                thresholds = api_info.get("congestion_thresholds", {})
                for level, info in thresholds.items():
                    st.write(f"**{level.capitalize()}**: {info}")
        
        st.markdown("---")
        st.caption("Built with Streamlit & FastAPI")
        
    return API_BASE_URL, api_info


def render_prediction_form():
    """Render the prediction input form."""
    st.subheader("🔮 Single Prediction")
    
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        selected_date = st.date_input(
            "Select Date",
            value=datetime.now(),
            min_value=datetime(2014, 1, 1),
            max_value=datetime(2030, 12, 31)
        )
        date_str = selected_date.strftime("%Y-%m-%d")
    
    with col2:
        selected_hour = st.slider("Hour of Day", 0, 23, 12)
    
    with col3:
        st.write("")  # Spacing
        st.write("")  # Spacing
        predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)
    
    return date_str, selected_hour, predict_btn


def render_prediction_results(prediction_data: Dict[str, Any], date_str: str):
    """Render prediction results with visualizations."""
    if not prediction_data or "prediction" not in prediction_data:
        st.error("No prediction data available")
        return
    
    pred = prediction_data["prediction"]
    
    # Metrics row
    st.subheader(f"📈 Prediction for {date_str}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        volume = pred["predicted_volume"]
        st.metric(
            "Predicted Volume",
            f"{volume:,}",
            help="Expected transaction volume"
        )
    
    with col2:
        level = pred["congestion_level"]
        level_colors = {
            "Low": "congestion-low",
            "Medium": "congestion-medium",
            "High": "congestion-high",
            "Critical": "congestion-critical"
        }
        st.markdown(
            f"<div class='metric-card'>"
            f"<h2 class='{level_colors.get(level, '')}'>{level}</h2>"
            f"<p>Congestion Level</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col3:
        pct = pred["congestion_percentage"]
        st.metric(
            "Congestion %",
            f"{pct:.1f}%",
            help="Congestion as percentage of max capacity"
        )
    
    with col4:
        conf = pred["confidence_score"]
        st.metric(
            "Confidence",
            f"{conf:.1%}",
            help="Prediction confidence score"
        )
    
    st.markdown("---")
    
    # Gauge visualization
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📉 Congestion Gauge")
        
        # Create gauge chart
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 25], 'color': '#28a745'},
                    {'range': [25, 50], 'color': '#ffc107'},
                    {'range': [50, 75], 'color': '#fd7e14'},
                    {'range': [75, 100], 'color': '#dc3545'}
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 4},
                    'thickness': 0.75,
                    'value': pct
                }
            }
        ))
        
        fig_gauge.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        st.plotly_chart(fig_gauge, use_container_width=True)
    
    with col2:
        st.subheader("📋 Recommendations")
        
        recommendations = pred.get("recommendations", [])
        for i, rec in enumerate(recommendations, 1):
            icon = "🔴" if "IMMEDIATE" in rec or "CRITICAL" in rec else "🟡" if "ALERT" in rec else "🟢"
            st.write(f"{icon} {rec}")
    
    st.markdown("---")


def render_time_series_prediction():
    """Render time series prediction section."""
    st.subheader("📅 Time Series Forecast")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now(),
            min_value=datetime(2014, 1, 1)
        )
    
    with col2:
        num_days = st.slider("Number of Days", 7, 90, 30)
    
    with col3:
        st.write("")
        st.write("")
        forecast_btn = st.button("📊 Generate Forecast", type="primary")
    
    if forecast_btn:
        dates = []
        features_list = []
        current_date = start_date
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(num_days):
            date_str = current_date.strftime("%Y-%m-%d")
            dates.append(date_str)
            features_list.append(build_features_from_date(date_str))
            current_date += timedelta(days=1)
            
            progress_bar.progress((i + 1) / num_days)
            status_text.text(f"Processing day {i + 1} of {num_days}...")
        
        status_text.text("Fetching predictions from API...")
        
        # Make batch prediction request
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict/batch",
                json={"features_list": features_list},
                timeout=API_TIMEOUT
            )
            
            if response.status_code == 200:
                data = response.json()
                predictions = data.get("predictions", [])
                
                # Create dataframe
                df = pd.DataFrame({
                    "Date": dates,
                    "Volume": [p["predicted_volume"] for p in predictions],
                    "Level": [p["congestion_level"] for p in predictions],
                    "Congestion %": [p["congestion_percentage"] for p in predictions]
                })
                
                progress_bar.empty()
                status_text.empty()
                
                # Display chart
                st.subheader(f"📈 Forecast: {num_days} Days Starting {start_date.strftime('%Y-%m-%d')}")
                
                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.1,
                    subplot_titles=("Transaction Volume Forecast", "Congestion Level"),
                    row_heights=[0.7, 0.3]
                )
                
                # Volume chart with color by level
                colors = {"Low": "#28a745", "Medium": "#ffc107", "High": "#fd7e14", "Critical": "#dc3545"}
                
                for level in ["Low", "Medium", "High", "Critical"]:
                    mask = df["Level"] == level
                    if mask.any():
                        fig.add_trace(
                            go.Scatter(
                                x=df[mask]["Date"],
                                y=df[mask]["Volume"],
                                mode='lines+markers',
                                name=level,
                                line=dict(color=colors[level]),
                                marker=dict(size=8)
                            ),
                            row=1, col=1
                        )
                
                # Congestion percentage
                fig.add_trace(
                    go.Scatter(
                        x=df["Date"],
                        y=df["Congestion %"],
                        mode='lines',
                        name="Congestion %",
                        fill='tozeroy',
                        fillcolor='rgba(31, 119, 180, 0.2)',
                        line=dict(color="#1f77b4", width=2)
                    ),
                    row=2, col=1
                )
                
                fig.update_layout(
                    height=500,
                    showlegend=True,
                    hovermode="x unified"
                )
                
                fig.update_yaxes(title_text="Volume", row=1, col=1)
                fig.update_yaxes(title_text="Congestion %", range=[0, 100], row=2, col=1)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                st.subheader("📊 Summary Statistics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Average Volume", f"{df['Volume'].mean():,.0f}")
                with col2:
                    st.metric("Max Volume", f"{df['Volume'].max():,}")
                with col3:
                    critical_days = (df["Level"] == "Critical").sum()
                    st.metric("Critical Days", critical_days)
                with col4:
                    high_days = (df["Level"] == "High").sum()
                    st.metric("High Congestion Days", high_days)
                
                # Data table
                with st.expander("📋 View Raw Data"):
                    st.dataframe(df, use_container_width=True)
            else:
                st.error(f"API Error: {response.text}")
        
        except Exception as e:
            st.error(f"Error: {str(e)}")


def render_historical_analysis():
    """Render historical pattern analysis."""
    st.subheader("📊 Pattern Analysis")
    
    # Day of week analysis
    st.write("### Transaction Patterns by Day of Week")
    
    # Generate sample data for visualization
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    dow_avg_volume = [6200, 6400, 6300, 6500, 6100, 3500, 3000]
    dow_critical_pct = [15, 20, 18, 25, 12, 5, 3]
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dow_labels,
            y=dow_avg_volume,
            marker_color=['#1f77b4'] * 5 + ['#28a745'] * 2,
            text=dow_avg_volume,
            textposition='auto'
        ))
        fig.update_layout(
            title="Average Volume by Day",
            yaxis_title="Volume",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dow_labels,
            y=dow_critical_pct,
            marker_color=['#dc3545' if p > 15 else '#fd7e14' if p > 10 else '#ffc107' for p in dow_critical_pct],
            text=[f"{p}%" for p in dow_critical_pct],
            textposition='auto'
        ))
        fig.update_layout(
            title="Critical Congestion Probability (%)",
            yaxis_title="Probability %",
            showlegend=False,
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Monthly analysis
    st.write("### Transaction Patterns by Month")
    
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_avg_volume = [4500, 4800, 5800, 5500, 5700, 6000, 5000, 5200, 5800, 6200, 6500, 5700]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=month_labels,
        y=month_avg_volume,
        mode='lines+markers',
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=10),
        text=month_avg_volume,
        textposition="top center"
    ))
    
    # Add threshold lines
    fig.add_hline(y=9000, line_dash="dot", annotation_text="Critical Threshold", line_color="#dc3545")
    fig.add_hline(y=6000, line_dash="dash", annotation_text="High Threshold", line_color="#fd7e14")
    fig.add_hline(y=3000, line_dash="dash", annotation_text="Medium Threshold", line_color="#ffc107")
    
    fig.update_layout(
        title="Monthly Average Transaction Volume",
        yaxis_title="Volume",
        showlegend=False,
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point."""
    render_header()
    
    # Render sidebar
    api_url, api_info = render_sidebar()
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📅 Forecast", "📊 Analysis"])
    
    with tab1:
        date_str, hour, predict_btn = render_prediction_form()
        
        if predict_btn:
            with st.spinner("Fetching prediction..."):
                prediction_data = get_prediction(date_str, hour)
                if prediction_data:
                    render_prediction_results(prediction_data, date_str)
    
    with tab2:
        render_time_series_prediction()
    
    with tab3:
        render_historical_analysis()
    
    # Footer
    st.markdown("---")
    st.caption(
        f"Transaction Congestion Predictor | API: {api_url} | "
        f"Model: congestion_model.joblib"
    )


if __name__ == "__main__":
    main()