from app.services.binance_service import binance_service
from app.ml.forecaster import price_forecaster

print("=" * 60)
print("TESTING CRYPTOCURRENCY PRICE FORECASTING")
print("=" * 60)

# Step 1: Get historical data (7 days = 168 hours)
print("\n📊 Fetching 7 days of Bitcoin historical data...")
historical_data = binance_service.get_historical_data("BTCUSDT", interval="1h", limit=168)
print(f"✓ Got {len(historical_data)} data points")

# Get current price for comparison
current_price = historical_data[-1]['close']
print(f"✓ Current BTC Price: ${current_price}")

print("\n" + "=" * 60)
print("RUNNING ALL 5 FORECASTING MODELS")
print("=" * 60)

# Test each model
models = ['arima', 'sarima', 'prophet', 'ets', 'ensemble']

for model_name in models:
    print(f"\n🔮 Testing {model_name.upper()} model...")
    
    result = price_forecaster.forecast(
        historical_data=historical_data,
        method=model_name,
        steps=24  # Forecast 24 hours ahead
    )
    
    if 'error' in result:
        print(f"   ❌ Error: {result['error']}")
    else:
        predicted = result['predicted_price']
        change = ((predicted - current_price) / current_price) * 100
        
        print(f"   ✓ Model: {result['model']}")
        print(f"   ✓ 24h Prediction: ${predicted:,.2f}")
        print(f"   ✓ Expected Change: {change:+.2f}%")
        
        if 'confidence_lower' in result:
            print(f"   ✓ Confidence Range: ${result['confidence_lower']:,.2f} - ${result['confidence_upper']:,.2f}")
        
        if model_name == 'ensemble':
            print(f"   ✓ Individual Predictions:")
            for name, pred in result['individual_predictions'].items():
                if pred:
                    print(f"      - {name}: ${pred:,.2f}")

print("\n" + "=" * 60)
print("TESTING COMPLETE!")
print("=" * 60)