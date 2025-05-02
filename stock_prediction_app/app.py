from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os
import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.utils as plotly_utils
import json
import pytz

app = Flask(__name__)

# Configure the models directory
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Stock portfolio mapping names to tickers AND model filenames
STOCK_DATA = {
    'Apple': {'ticker': 'AAPL', 'model_file': 'apple_model.pkl'},
    'Microsoft': {'ticker': 'MSFT', 'model_file': 'msft_model.pkl'},
    'Amazon': {'ticker': 'AMZN', 'model_file': 'amazon_model.pkl'},
    'Google': {'ticker': 'GOOGL', 'model_file': 'google_model.pkl'},
    'Meta': {'ticker': 'META', 'model_file': 'meta_model.pkl'},
    'Tesla': {'ticker': 'TSLA', 'model_file': 'tesla_model.pkl'},
    'NVIDIA': {'ticker': 'NVDA', 'model_file': 'nvda_model.pkl'},
    'JPMorgan': {'ticker': 'JPM', 'model_file': 'jpm_model.pkl'},
    'Visa': {'ticker': 'V', 'model_file': 'v_model.pkl'},
    'Walmart': {'ticker': 'WMT', 'model_file': 'wmt_model.pkl'}
}

# Load all pre-trained models
stock_models = {}
available_stocks = []

for stock_name, data in STOCK_DATA.items():
    model_path = os.path.join(MODELS_DIR, data['model_file'])
    try:
        if os.path.exists(model_path):
            stock_models[stock_name] = joblib.load(model_path)
            available_stocks.append((stock_name, data['ticker']))
            print(f"✅ Loaded model for {stock_name} ({data['ticker']})")
        else:
            print(f"❌ Model file not found: {model_path}")
    except Exception as e:
        print(f"❌ Error loading model for {stock_name}: {str(e)}")

def get_current_stock_data(ticker):
    """Fetch current stock data including moving averages"""
    try:
        # Get data for last 60 days to calculate moving averages
        data = yf.download(ticker, period="60d")
        
        if data.empty or 'Close' not in data.columns:
            print("⚠️ No valid data returned from Yahoo Finance!")
            return None
            
        # Calculate indicators
        data['MA_10'] = data['Close'].rolling(10).mean()
        data['MA_50'] = data['Close'].rolling(50).mean()
        
        # Get the latest data point
        latest = data.iloc[-1]
        
        return {
        'current_price': float(data['Close'].iloc[-1]),  
        'ma10': float(data['MA_10'].iloc[-1]),          
        'ma50': float(data['MA_50'].iloc[-1]),          
        'last_updated': data.index[-1].strftime('%Y-%m-%d')
    }
        
    except Exception as e:
        print(f"❌ Error fetching stock data: {str(e)}")
        return None

def generate_historical_test_cases(ticker, start_date, end_date):
    """Generate test cases from real historical data"""
    print(f"\nFetching data for {ticker} from {start_date} to {end_date}")
    
    try:
        # Download data with buffer for moving averages
        data = yf.download(ticker, 
                          start=start_date - timedelta(days=60),
                          end=end_date)
        
        if data.empty or 'Close' not in data.columns:
            print("⚠️ No valid data returned from Yahoo Finance!")
            return []
            
        # Calculate indicators
        data['MA_10'] = data['Close'].rolling(10).mean()
        data['MA_50'] = data['Close'].rolling(50).mean()
        
        # Filter to requested date range and clean
        data = data.loc[start_date:end_date].dropna()
        
        if len(data) < 2:  # Need at least 2 days for prediction
            print("⚠️ Insufficient data after cleaning")
            return []
        
        print(f"Processed {len(data)} rows of data")
        
        # Generate test cases
        test_cases = []
        for i in range(len(data)-1):
            test_cases.append({
                'date': data.index[i].strftime('%Y-%m-%d'),
                'current_price': float(data['Close'].iloc[i]),
                'ma10': float(data['MA_10'].iloc[i]),
                'ma50': float(data['MA_50'].iloc[i]),
                'actual_next_day': float(data['Close'].iloc[i+1])
            })
        
        print(f"Generated {len(test_cases)} test cases")
        return test_cases
        
    except Exception as e:
        print(f"❌ Error generating test cases: {str(e)}")
        return []

def calculate_direction_accuracy(results):
    """Calculate percentage of correct direction predictions"""
    if len(results) < 2:
        return 0.0
    
    correct = 0
    for i in range(1, len(results)):
        pred_direction = np.sign(results[i]['predicted'] - results[i-1]['actual'])
        actual_direction = np.sign(results[i]['actual'] - results[i-1]['actual'])
        
        if pred_direction == actual_direction:
            correct += 1
    
    return round(correct / (len(results)-1) * 100, 2)

def get_prediction_date():
    """Get the next trading day (skips weekends) with proper timezone handling"""
    # Get current time in market timezone (NYSE in this case)
    ny_tz = pytz.timezone('America/New_York')
    now = datetime.now(ny_tz)
    
    # Calculate next day
    next_day = now + timedelta(days=1)
    
    # Adjust for weekends
    if next_day.weekday() >= 5:  # Saturday (5) or Sunday (6)
        next_day += timedelta(days=(7 - next_day.weekday()))
    
    # Format as YYYY-MM-DD
    return next_day.strftime('%Y-%m-%d')

def create_price_graph(stock_name, ticker, current_price, predicted_price):
    """Create a focused prediction graph between current and predicted prices"""
    # Get minimal historical data (just 5 days for context)
    data = yf.download(ticker, period="5d")
    
    if data.empty:
        return None
    
    # Create figure
    fig = go.Figure()
    
    # 1. Thin line for recent historical context
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'],
        mode='lines',
        name='Recent Prices',
        line=dict(color='#888', width=1),
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    # Calculate prediction date
    next_day = data.index[-1] + pd.Timedelta(days=1)
    if next_day.weekday() >= 5:  # Handle weekends
        next_day += pd.Timedelta(days=(7 - next_day.weekday()))
    
    # 2. Current price marker
    fig.add_trace(go.Scatter(
        x=[data.index[-1]],
        y=[current_price],
        mode='markers+text',
        name='Current Price',
        marker=dict(color='#0F9D58', size=14),
        text=['NOW'],
        textposition="top center",
        hovertemplate='Current: %{y:.2f}<extra></extra>',
        textfont=dict(size=12, color='#0F9D58')
    ))
    
    # 3. Prediction bridge (shaded area between current and predicted)
    fig.add_trace(go.Scatter(
        x=[data.index[-1], next_day],
        y=[current_price, predicted_price],
        fill='tozeroy',
        mode='none',
        name='Prediction Range',
        fillcolor='rgba(234, 67, 53, 0.2)',
        showlegend=False
    ))
    
    # 4. Prediction line
    fig.add_trace(go.Scatter(
        x=[data.index[-1], next_day],
        y=[current_price, predicted_price],
        mode='lines',
        line=dict(color='#DB4437', width=3),
        name='Predicted Change',
        hovertemplate='%{y:.2f}<extra></extra>'
    ))
    
    # 5. Predicted price marker
    fig.add_trace(go.Scatter(
        x=[next_day],
        y=[predicted_price],
        mode='markers+text',
        name='Predicted Price',
        marker=dict(color='#DB4437', size=14),
        text=['TOMORROW'],
        textposition="top center",
        hovertemplate='Predicted: %{y:.2f}<extra></extra>',
        textfont=dict(size=12, color='#DB4437')
    ))
    
    # Calculate price change
    change = predicted_price - current_price
    pct_change = (change / current_price) * 100
    
    # Layout configuration
    fig.update_layout(
        title={
            'text': f"{stock_name} ({ticker}) Price Prediction",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'font': dict(size=18)
        },
        xaxis=dict(
            title="Timeline",
            showgrid=False,
            range=[data.index[-1] - pd.Timedelta(hours=12), 
                     next_day + pd.Timedelta(hours=12)]
        ),
        yaxis=dict(
            title="Price ($)",
            range=[min(current_price, predicted_price)*0.995, 
                   max(current_price, predicted_price)*1.005],
            tickprefix="$"
        ),
        template="plotly_white",
        hovermode="x unified",
        showlegend=False,
        margin=dict(l=50, r=50, b=80, t=100),
        annotations=[
            dict(
                x=next_day,
                y=predicted_price,
                text=f"<b>{'↑' if change >=0 else '↓'} ${predicted_price:.2f}</b><br>{'+' if change >=0 else ''}{change:.2f} ({'+' if pct_change >=0 else ''}{pct_change:.2f}%)",
                showarrow=True,
                arrowhead=3,
                ax=0,
                ay=-40,
                font=dict(size=12, color='#DB4437'),
                bordercolor="#DB4437",
                borderwidth=1,
                borderpad=4,
                bgcolor="white"
            )
        ]
    )
    
    # Custom styling
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black')
    
    return json.dumps(fig, cls=plotly_utils.PlotlyJSONEncoder)

@app.route('/predict_data')
def predict_data():
    # Only show stocks with loaded models
    available_stocks = [name for name in STOCK_DATA if name in stock_models]
    return render_template('index.html', stocks=available_stocks)

@app.route('/')
def index():
    # Only show stocks with loaded models
    available_stocks = [name for name in STOCK_DATA if name in stock_models]
    return render_template('landing_page.html')

@app.route('/predict_form')
def predict_form():
    """Simplified prediction form with automatic data fetching"""
    return render_template('predict_form.html', available_stocks=available_stocks)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction with automatic data fetching"""
    stock_name = request.form['stock_name']
    quantity = int(request.form.get('quantity', 1))
    
    if stock_name not in stock_models:
        return render_template('error.html',
                            message=f"No model loaded for {stock_name}"), 404
    
    ticker = STOCK_DATA[stock_name]['ticker']
    
    # Fetch current stock data
    stock_data = get_current_stock_data(ticker)
    if not stock_data:
        return render_template('error.html',
                            message=f"Could not fetch current data for {stock_name}"), 400
    
    # Prepare input features as numpy array (without feature names)
    features = np.array([
        [float(stock_data['current_price']),
         float(stock_data['ma10']),
         float(stock_data['ma50'])]
    ])
    
    # Make prediction
    try:
        predicted_price = stock_models[stock_name].predict(features)[0]
        predicted_change = predicted_price - float(stock_data['current_price'])
        predicted_pct_change = (predicted_change / float(stock_data['current_price'])) * 100
        
        # Calculate position value
        current_value = float(stock_data['current_price']) * quantity
        predicted_value = predicted_price * quantity
        value_change = predicted_value - current_value
        
        # Generate prediction date (next business day)
        today = datetime.now()
        next_day = today + timedelta(days=1)
        if next_day.weekday() >= 5:  # If Saturday or Sunday
            next_day += timedelta(days=(7 - next_day.weekday()))
        
        # Create price graph
        graph_json = create_price_graph(stock_name, ticker, 
                                      float(stock_data['current_price']), 
                                      predicted_price)
        
        return render_template('prediction_results.html',
                            stock_name=stock_name,
                            ticker=ticker,
                            current_price=round(float(stock_data['current_price']), 2),
                            ma10=round(float(stock_data['ma10']), 2),
                            ma50=round(float(stock_data['ma50']), 2),
                            last_updated=stock_data['last_updated'],
                            predicted_price=round(predicted_price, 2),
                            predicted_change=round(predicted_change, 2),
                            predicted_pct_change=round(predicted_pct_change, 2),
                            prediction_date=get_prediction_date(),
                            quantity=quantity,
                            current_value=round(current_value, 2),
                            predicted_value=round(predicted_value, 2),
                            value_change=round(value_change, 2),
                            graph_json=graph_json)
        
    except Exception as e:
        return render_template('error.html',
                            message=f"Prediction failed: {str(e)}"), 500

@app.route('/select_stock')
def select_stock():
    """Stock selection page with date range picker"""
    today = datetime.now().date()
    default_end = today - timedelta(days=1)
    default_start = default_end - timedelta(days=30)
    
    return render_template('select_stock.html',
                        available_stocks=available_stocks,
                        default_start=default_start.strftime('%Y-%m-%d'),
                        default_end=default_end.strftime('%Y-%m-%d'),
                        today=today.strftime('%Y-%m-%d'))

@app.route('/test_selected_stock', methods=['POST'])
def test_selected_stock():
    """Run accuracy test for selected stock and date range"""
    if not stock_models:
        return render_template('error.html',
                            message="No models loaded. Please check your model files."), 500
    
    stock_name = request.form['stock_name']
    ticker = request.form['ticker']
    
    try:
        start_date = datetime.strptime(request.form['start_date'], '%Y-%m-%d').date()
        end_date = datetime.strptime(request.form['end_date'], '%Y-%m-%d').date()
        today = datetime.now().date()
        
        if start_date >= end_date:
            return render_template('error.html',
                                message="End date must be after start date"), 400
        if end_date > today:
            return render_template('error.html',
                                message="Cannot test future dates"), 400
            
    except ValueError:
        return render_template('error.html',
                            message="Invalid date format"), 400
    
    # Verify model exists
    if stock_name not in stock_models:
        return render_template('error.html',
                            message=f"No model loaded for {stock_name}"), 404
    
    # Get test cases
    test_cases = generate_historical_test_cases(ticker, start_date, end_date)
    if not test_cases:
        return render_template('error.html',
                            message=f"No historical data found for {stock_name} between {start_date} and {end_date}"), 400
    
    # Run predictions
    results = []
    for case in test_cases:
        try:
            features = np.array([
                [case['current_price']],
                [case['ma10']],
                [case['ma50']]
            ]).reshape(1, -1)
            
            predicted = stock_models[stock_name].predict(features)[0]
            
            results.append({
                'date': case['date'],
                'predicted': round(predicted, 2),
                'actual': round(case['actual_next_day'], 2),
                'error': round(abs(predicted - case['actual_next_day']), 2),
                'percent_error': round(abs(predicted - case['actual_next_day'])/case['actual_next_day']*100, 2)
            })
        except Exception as e:
            print(f"❌ Prediction error for {case['date']}: {str(e)}")
            continue
    
    if not results:
        return render_template('error.html',
                            message="No successful predictions generated"), 400
    
    # Calculate statistics
    stats = {
        'stock_name': stock_name,
        'ticker': ticker,
        'start_date': start_date.strftime('%Y-%m-%d'),
        'end_date': end_date.strftime('%Y-%m-%d'),
        'test_days': len(results),
        'avg_error': round(np.mean([r['error'] for r in results]), 2),
        'accuracy': round(100 - np.mean([r['percent_error'] for r in results]), 2),
        'direction_accuracy': calculate_direction_accuracy(results)
    }
    
    print(f"✅ Test completed for {stock_name}. Results: {stats}")
    return render_template('stock_test_results.html',
                        stats=stats,
                        results=results[:100])  # Show first 100 results

if __name__ == '__main__':
    # Verify models loaded
    print("\nLoaded models:")
    for stock, model in stock_models.items():
        print(f"- {stock}: {model.__class__.__name__}")
    
    app.run(debug=True)