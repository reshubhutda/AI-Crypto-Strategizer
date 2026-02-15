import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

class MonteCarloSimulator:
    """
    Monte Carlo Simulation for Cryptocurrency Risk Analysis
    
    Uses Geometric Brownian Motion (GBM) to simulate thousands of 
    possible price paths and calculate risk metrics.
    
    KEY CONCEPTS:
    - Geometric Brownian Motion: Models random price movements
    - Value at Risk (VaR): Maximum expected loss at confidence level
    - Probability Distribution: Range of possible outcomes
    - Risk/Reward Ratio: Potential gain vs potential loss
    """
    
    def __init__(self):
        logger.info("Monte Carlo Simulator initialized")
    
    def calculate_volatility(self, historical_data: List[Dict]) -> float:
        """
        Calculate historical volatility (standard deviation of returns)
        
        CORRECTED: Properly handles hourly crypto data
        """
        try:
            # Extract close prices
            prices = [float(candle['close']) for candle in historical_data]
            
            # Calculate log returns (more accurate for percentages)
            log_returns = []
            for i in range(1, len(prices)):
                log_ret = np.log(prices[i] / prices[i-1])
                log_returns.append(log_ret)
            
            # Hourly volatility
            hourly_volatility = np.std(log_returns)
            
            # Annualize for hourly data: vol * sqrt(hours per year)
            # BUT we want to use it for hourly simulation, so we keep it hourly!
            # Only annualize for display purposes
            annualized_vol = hourly_volatility * np.sqrt(24 * 365)
            
            logger.info(f"Hourly volatility: {hourly_volatility:.6f}")
            logger.info(f"Annualized volatility: {annualized_vol:.4f}")
            
            # Return HOURLY volatility (not annualized) for simulation
            return hourly_volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            # Default to reasonable hourly volatility
            return 0.015  # ~1.5% hourly = ~65% annualized (more reasonable for crypto)
    
    def simulate_price_paths(
        self,
        current_price: float,
        volatility: float,
        days: int = 7,
        simulations: int = 10000,
        drift: float = 0.0
    ) -> np.ndarray:
        """
        Simulate future price paths using Geometric Brownian Motion
        
        FORMULA:
        S(t+1) = S(t) * exp((μ - σ²/2)*dt + σ*sqrt(dt)*Z)
        
        Where:
        - S(t) = price at time t
        - μ = drift (expected return, usually 0 for neutral)
        - σ = volatility
        - dt = time step (1 hour)
        - Z = random shock from normal distribution
        
        Args:
            current_price: Starting price
            volatility: Annual volatility
            days: Number of days to simulate
            simulations: Number of paths to generate
            drift: Expected return (0 = no trend assumption)
        
        Returns:
            Array of shape (simulations, time_steps) with price paths
        """
        logger.info(f"Simulating {simulations:,} price paths for {days} days...")
        
        # Time parameters
        dt = 1/24  # Hourly steps (1/24 of a day)
        time_steps = days * 24
        
        # Initialize array to store all paths
        price_paths = np.zeros((simulations, time_steps))
        
        # Run simulations
        for i in range(simulations):
            prices = [current_price]
            
            for step in range(time_steps - 1):
                # Generate random shock from normal distribution
                random_shock = np.random.normal(0, 1)
                
                # Geometric Brownian Motion formula
                price_change = prices[-1] * np.exp(
                    (drift - 0.5 * volatility**2) * dt +
                    volatility * np.sqrt(dt) * random_shock
                )
                
                prices.append(price_change)
            
            price_paths[i] = prices
        
        logger.info("Simulation complete!")
        return price_paths
    
    def calculate_metrics(
        self,
        price_paths: np.ndarray,
        current_price: float,
        confidence_level: float = 0.95
    ) -> Dict:
        """
        Calculate risk metrics from simulated paths
        
        Args:
            price_paths: Simulated price paths
            current_price: Starting price
            confidence_level: Confidence level for VaR (default 95%)
        
        Returns:
            Dictionary with risk metrics
        """
        # Get final prices from all simulations
        final_prices = price_paths[:, -1]
        
        # Calculate statistics
        mean_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        std_dev = np.std(final_prices)
        
        # Percentiles for confidence intervals
        percentile_5 = np.percentile(final_prices, 5)    # 5% worst case
        percentile_25 = np.percentile(final_prices, 25)  # 25% worst case
        percentile_75 = np.percentile(final_prices, 75)  # 75% best case
        percentile_95 = np.percentile(final_prices, 95)  # 95% best case
        
        # Value at Risk (VaR)
        var_5 = current_price - percentile_5  # Loss in worst 5% of cases
        
        # Probability of profit
        profitable_paths = np.sum(final_prices > current_price)
        probability_profit = profitable_paths / len(final_prices)
        
        # Expected return
        expected_return = (mean_price - current_price) / current_price
        
        return {
            "mean_price": float(mean_price),
            "median_price": float(median_price),
            "std_dev": float(std_dev),
            "percentile_5": float(percentile_5),
            "percentile_25": float(percentile_25),
            "percentile_75": float(percentile_75),
            "percentile_95": float(percentile_95),
            "value_at_risk_5pct": float(var_5),
            "probability_profit": float(probability_profit),
            "expected_return": float(expected_return),
            "all_final_prices": final_prices.tolist()
        }
    
    def calculate_optimal_levels(
        self,
        current_price: float,
        simulation_results: Dict,
        risk_tolerance: float = 1.0
    ) -> Dict:
        """
        Calculate optimal entry, exit, and stop-loss levels
        
        Args:
            current_price: Current market price
            simulation_results: Results from Monte Carlo simulation
            risk_tolerance: Multiplier for risk (1.0 = normal, 0.5 = conservative, 2.0 = aggressive)
        
        Returns:
            Optimal trading levels
        """
        # Conservative entry: Wait for dip to 25th percentile
        optimal_entry = simulation_results['percentile_25'] * 0.98
        
        # Target exit: Aim for median predicted price
        optimal_exit = simulation_results['median_price']
        
        # Stop loss: Based on Value at Risk
        var = simulation_results['value_at_risk_5pct']
        stop_loss = current_price - (var * risk_tolerance)
        
        # Calculate risk/reward ratio
        potential_gain = optimal_exit - optimal_entry
        potential_loss = optimal_entry - stop_loss
        
        if potential_loss > 0:
            risk_reward_ratio = potential_gain / potential_loss
        else:
            risk_reward_ratio = 0
        
        return {
            "optimal_entry": float(optimal_entry),
            "optimal_exit": float(optimal_exit),
            "stop_loss": float(stop_loss),
            "potential_gain": float(potential_gain),
            "potential_loss": float(potential_loss),
            "risk_reward_ratio": float(risk_reward_ratio)
        }
    
    def run_simulation(
        self,
        current_price: float,
        historical_data: List[Dict],
        days: int = 7,
        simulations: int = 10000,
        risk_tolerance: float = 1.0
    ) -> Dict:
        """Main method: Run complete Monte Carlo analysis"""
        try:
            # Step 1: Calculate volatility
            hourly_volatility = self.calculate_volatility(historical_data)
            annualized_volatility = hourly_volatility * np.sqrt(24 * 365)
            
            # Step 2: Run simulations (use hourly volatility)
            price_paths = self.simulate_price_paths(
                current_price=current_price,
                volatility=hourly_volatility,  # ← Use hourly, not annualized!
                days=days,
                simulations=simulations
            )
            
            # Step 3: Calculate metrics
            metrics = self.calculate_metrics(
                price_paths=price_paths,
                current_price=current_price
            )
            
            # Step 4: Calculate optimal levels
            optimal_levels = self.calculate_optimal_levels(
                current_price=current_price,
                simulation_results=metrics,
                risk_tolerance=risk_tolerance
            )
            
            # Combine results
            return {
                "current_price": float(current_price),
                "hourly_volatility": float(hourly_volatility),
                "annualized_volatility": float(annualized_volatility),
                "days_simulated": days,
                "num_simulations": simulations,
                "risk_tolerance": float(risk_tolerance),
                "metrics": metrics,
                "optimal_levels": optimal_levels,
                "recommendation": self._generate_recommendation(metrics, optimal_levels)
            }
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {e}")
            return {"error": str(e)}
    
    def _generate_recommendation(self, metrics: Dict, levels: Dict) -> str:
        """Generate human-readable recommendation"""
        prob = metrics['probability_profit']
        rr_ratio = levels['risk_reward_ratio']
        
        if prob > 0.7 and rr_ratio > 2:
            return "STRONG BUY - High probability of profit with good risk/reward"
        elif prob > 0.6 and rr_ratio > 1.5:
            return "BUY - Favorable odds and acceptable risk"
        elif prob > 0.5:
            return "HOLD - Slightly positive but risky"
        elif prob > 0.4:
            return "CAUTION - Low probability of profit"
        else:
            return "AVOID - High risk of loss"

# Create singleton
monte_carlo_simulator = MonteCarloSimulator()