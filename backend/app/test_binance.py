from services.binance_service import binance_service

# Test 1: Get BTC price
print("Testing get_price...")
try:
    price = binance_service.get_price("BTCUSDT")
    print(f"✓ BTC Price: ${price['price']}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Get historical data
print("\nTesting get_historical_data...")
try:
    data = binance_service.get_historical_data("BTCUSDT", interval="1h", limit=5)
    print(f"✓ Got {len(data)} candles")
    print(f"✓ Latest close price: ${data[-1]['close']}")
except Exception as e:
    print(f"✗ Error: {e}")