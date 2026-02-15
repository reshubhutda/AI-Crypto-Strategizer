from app.services.binance_service import binance_service
from app.ml.monte_carlo import monte_carlo_simulator

print("=" * 70)
print("MONTE CARLO RISK ANALYSIS - CRYPTOCURRENCY")
print("=" * 70)

# Get current price and historical data
print("\n📊 Fetching Bitcoin data...")
current_data = binance_service.get_price("BTCUSDT")
current_price = float(current_data['price'])
print(f"✓ Current BTC Price: ${current_price:,.2f}")

historical_data = binance_service.get_historical_data("BTCUSDT", "1h", 168)
print(f"✓ Got {len(historical_data)} hours of historical data")

print("\n" + "=" * 70)
print("RUNNING MONTE CARLO SIMULATION (10,000 scenarios)")
print("=" * 70)

# Run simulation
result = monte_carlo_simulator.run_simulation(
    current_price=current_price,
    historical_data=historical_data,
    days=7,
    simulations=10000,
    risk_tolerance=1.0
)

if 'error' in result:
    print(f"\n❌ Error: {result['error']}")
else:
    print(f"\n📈 SIMULATION RESULTS:")
    print(f"   Hourly Volatility: {result['hourly_volatility']*100:.2f}%")
    print(f"   Annualized Volatility: {result['annualized_volatility']*100:.2f}%")
    print(f"   Simulations: {result['num_simulations']:,}")
    print(f"   Time Horizon: {result['days_simulated']} days")
    
    metrics = result['metrics']
    print(f"\n📊 PRICE PREDICTIONS (7 days):")
    print(f"   Mean Prediction: ${metrics['mean_price']:,.2f}")
    print(f"   Median Prediction: ${metrics['median_price']:,.2f}")
    print(f"   Standard Deviation: ${metrics['std_dev']:,.2f}")
    
    print(f"\n📉 RISK ANALYSIS:")
    print(f"   Best Case (95th percentile): ${metrics['percentile_95']:,.2f}")
    print(f"   Likely Range (25th-75th): ${metrics['percentile_25']:,.2f} - ${metrics['percentile_75']:,.2f}")
    print(f"   Worst Case (5th percentile): ${metrics['percentile_5']:,.2f}")
    print(f"   Value at Risk (VaR 5%): ${metrics['value_at_risk_5pct']:,.2f}")
    
    print(f"\n💰 PROBABILITIES:")
    print(f"   Probability of Profit: {metrics['probability_profit']*100:.1f}%")
    print(f"   Expected Return: {metrics['expected_return']*100:+.2f}%")
    
    levels = result['optimal_levels']
    print(f"\n🎯 OPTIMAL TRADING LEVELS:")
    print(f"   Suggested Entry: ${levels['optimal_entry']:,.2f}")
    print(f"   Target Exit: ${levels['optimal_exit']:,.2f}")
    print(f"   Stop Loss: ${levels['stop_loss']:,.2f}")
    print(f"   Potential Gain: ${levels['potential_gain']:,.2f} ({(levels['potential_gain']/current_price)*100:.2f}%)")
    print(f"   Potential Loss: ${levels['potential_loss']:,.2f} ({(levels['potential_loss']/current_price)*100:.2f}%)")
    print(f"   Risk/Reward Ratio: {levels['risk_reward_ratio']:.2f}:1")
    
    print(f"\n💡 RECOMMENDATION: {result['recommendation']}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE!")
print("=" * 70)