from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import os

app = Flask(__name__)

# Configure the models directory
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
            # PROPER 2D ARRAY FORMAT - THIS IS THE CRITICAL FIX
            features = np.array([
                [case['current_price']],  # First feature
                [case['ma10']],          # Second feature
                [case['ma50'] ]          # Third feature
            ]).reshape(1, -1)  # Reshape to (1, 3)
            
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

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    stock_name = request.form['stock']
    current_price = float(request.form['current_price'])
    ma10 = float(request.form['ma10'])
    ma50 = float(request.form['ma50'])
    
    # Prepare input features
    features = np.array([[current_price, ma10, ma50]])
    
    # Make prediction
    model = stock_models[stock_name]
    prediction = model.predict(features)[0]
    
    # Generate prediction date (next business day)
    today = datetime.now()
    next_day = today + timedelta(days=1)
    if next_day.weekday() >= 5:  # If Saturday or Sunday
        next_day += timedelta(days=(7 - next_day.weekday()))
    
    return render_template('results.html',
                         stock=stock_name,
                         current_price=current_price,
                         prediction=round(prediction, 2),
                         prediction_date=next_day.strftime('%Y-%m-%d'))

if __name__ == '__main__':
    # Verify models loaded
    print("\nLoaded models:")
    for stock, model in stock_models.items():
        print(f"- {stock}: {model.__class__.__name__}")
    
    app.run(debug=True)