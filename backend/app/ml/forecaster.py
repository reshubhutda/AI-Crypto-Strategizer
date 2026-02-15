import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
try:
    from pmdarima import auto_arima
    USE_AUTO_ARIMA = True
except:
    USE_AUTO_ARIMA = False
import logging
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PriceForecaster:
    """
    ulti-model time series forecasting for cryptocurrency prices
    
    Implements 5 forecasting methods:
    1. ARIMA - Classic statistical forecasting
    2. SARIMA - ARIMA with seasonality
    3. Prophet - Facebook's robust forecaster
    4. Exponential Smoothing - Weighted recent data
    5. Ensemble - Combines all models
    """
    def __init__(self):
        self.model = {}
        logger.info("Price Forecaster Initialized")
    
    def prepare_data(self, historical_data: List[Dict]) -> pd.DataFrame:
        """
        Convert historical data to pandas DataFrame
        
        Args:
            historical_data: List of OHLCV data from Binance
        
        Returns:
            DataFrame with timestamp and close price
        """
        df = pd.DataFrame(historical_data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df.set_index('timestamp')
        
        # Use close price for forecasting
        df['price'] = df['close'].astype(float)
        
        return df[['price']]
    
    # ARIMA #

    def forecast_arima(self, data: pd.DataFrame, steps: int = 24) -> Dict:
        """
        ARIMA: Auto-Regressive Integrated Moving Average
        
        HOW IT WORKS:
        - AR (p): Uses past values to predict future
        - I (d): Differences data to make it stationary
        - MA (q): Uses past forecast errors
        
        BEST FOR: Short-term linear trends
        """
        try:
            logger.info("Running ARIMA forecast...")
            
            if USE_AUTO_ARIMA:
                try:
                    # Use auto_arima with more robust settings
                    model = auto_arima(
                        data['price'],
                        start_p=0, start_q=0,  # Start from 0
                        max_p=3, max_q=3,      # Reduced complexity
                        d=None,                # Auto-detect differencing
                        max_d=2,               # Max differencing
                        seasonal=False,
                        trace=False,
                        error_action='ignore',
                        suppress_warnings=True,
                        stepwise=True,
                        n_jobs=-1,             # Use all CPU cores
                        maxiter=50,            # Limit iterations
                        method='lbfgs'         # More stable optimization
                    )
                    
                    model.fit(data['price'])
                    forecast, conf_int = model.predict(n_periods=steps, return_conf_int=True)
                    
                    return {
                        "model": "ARIMA",
                        "predicted_price": float(forecast[-1]),
                        "forecast_24h": float(forecast[-1]),
                        "confidence_lower": float(conf_int[-1][0]),
                        "confidence_upper": float(conf_int[-1][1]),
                        "all_forecasts": forecast.tolist(),
                        "parameters": str(model.order)
                    }
                except Exception as auto_error:
                    logger.warning(f"auto_arima failed: {auto_error}, falling back to manual ARIMA")
                    # Fall through to manual ARIMA below
            
            # Fallback: Manual ARIMA with conservative parameters
            logger.info("Using manual ARIMA with order (1,1,1)")
            model = ARIMA(data['price'], order=(1, 1, 1))
            fitted_model = model.fit()
            forecast_obj = fitted_model.get_forecast(steps=steps)
            forecast = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int()
            
            return {
                "model": "ARIMA_Manual",
                "predicted_price": float(forecast.iloc[-1]),
                "forecast_24h": float(forecast.iloc[-1]),
                "confidence_lower": float(conf_int.iloc[-1, 0]),
                "confidence_upper": float(conf_int.iloc[-1, 1]),
                "all_forecasts": forecast.tolist()
            }
            
        except Exception as e:
            logger.error(f"ARIMA error: {e}")
            return {"model": "ARIMA", "error": str(e)}
    

    # SARIMA #

    def forecast_sarima(self, data: pd.DataFrame, steps: int = 24) -> Dict:
        """
        SARIMA: Seasonal ARIMA
        
        HOW IT WORKS:
        - Same as ARIMA but adds seasonal component
        - Detects repeating patterns (daily, weekly)
        - Crypto often has weekend/weekday patterns
        
        BEST FOR: Data with recurring cycles
        
        Args:
            data: Historical price data
            steps: Hours to forecast
        
        Returns:
            Forecast with seasonality considered
        """
        try:
            logger.info("Running SARIMA forecast...")
            
            # For hourly data, season = 24 (daily cycle)
            model = SARIMAX(
                data['price'],
                order=(2, 1, 2),  # ARIMA parameters
                seasonal_order=(1, 1, 1, 24),  # (P, D, Q, season)
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            fitted_model = model.fit(disp=False)
            
            # Forecast
            forecast = fitted_model.forecast(steps=steps)
            
            # Confidence intervals
            forecast_df = fitted_model.get_forecast(steps=steps)
            conf_int = forecast_df.conf_int()
            
            return {
                "model": "SARIMA",
                "predicted_price": float(forecast.iloc[-1]),
                "forecast_24h": float(forecast.iloc[-1]),
                "confidence_lower": float(conf_int.iloc[-1, 0]),
                "confidence_upper": float(conf_int.iloc[-1, 1]),
                "all_forecasts": forecast.tolist()
            }
            
        except Exception as e:
            logger.error(f"SARIMA error: {e}")
            return {"model": "SARIMA", "error": str(e)}
    
    # PROPHET #

    def forecast_prophet(self, data: pd.DataFrame, steps: int = 24) -> Dict:
        """
        Prophet: Facebook's Forecasting Tool
        
        HOW IT WORKS:
        - Decomposes time series into: trend + seasonality + holidays
        - Handles missing data automatically
        - Robust to outliers
        
        BEST FOR: Long-term forecasts, messy data
        
        Args:
            data: Historical price data
            steps: Hours to forecast
        
        Returns:
            Robust forecast with uncertainty
        """
        try:
            logger.info("Running Prophet forecast...")
            
            # Prophet requires specific column names: 'ds' and 'y'
            prophet_df = pd.DataFrame({
                'ds': data.index,
                'y': data['price'].values
            })
            
            # Initialize model
            model = Prophet(
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False,  # Not enough data usually
                changepoint_prior_scale=0.05,  # Flexibility of trend
                interval_width=0.95  # 95% confidence interval
            )
            
            # Fit
            model.fit(prophet_df)
            
            # Create future dataframe
            future = model.make_future_dataframe(periods=steps, freq='H')
            
            # Predict
            forecast = model.predict(future)
            
            # Get last prediction
            last_forecast = forecast.iloc[-1]
            
            return {
                "model": "Prophet",
                "predicted_price": float(last_forecast['yhat']),
                "forecast_24h": float(last_forecast['yhat']),
                "confidence_lower": float(last_forecast['yhat_lower']),
                "confidence_upper": float(last_forecast['yhat_upper']),
                "trend": float(last_forecast['trend']),
                "all_forecasts": forecast['yhat'].tail(steps).tolist()
            }
            
        except Exception as e:
            logger.error(f"Prophet error: {e}")
            return {"model": "Prophet", "error": str(e)}
        
    # EXPONENTIAL SMOOTHNING #

    def forecast_exponential_smoothing(self, data: pd.DataFrame, steps: int = 24) -> Dict:
        """
        Exponential Smoothing (ETS)
        
        HOW IT WORKS:
        - Gives more weight to recent data
        - Simple but effective for short-term
        - Smooth predictions
        
        BEST FOR: Fast, simple forecasts
        
        Args:
            data: Historical price data
            steps: Hours to forecast
        
        Returns:
            Smoothed forecast
        """
        try:
            logger.info("Running Exponential Smoothing forecast...")
            
            # ETS model (Error, Trend, Seasonal)
            model = ExponentialSmoothing(
                data['price'],
                trend='add',  # Additive trend
                seasonal='add',  # Additive seasonality
                seasonal_periods=24  # Daily cycle
            )
            
            fitted_model = model.fit()
            
            # Forecast
            forecast = fitted_model.forecast(steps=steps)
            
            return {
                "model": "Exponential_Smoothing",
                "predicted_price": float(forecast.iloc[-1]),
                "forecast_24h": float(forecast.iloc[-1]),
                "all_forecasts": forecast.tolist()
            }
            
        except Exception as e:
            logger.error(f"Exponential Smoothing error: {e}")
            # Fallback to simple exponential smoothing
            try:
                model = ExponentialSmoothing(data['price'], trend='add')
                fitted_model = model.fit()
                forecast = fitted_model.forecast(steps=steps)
                return {
                    "model": "Exponential_Smoothing_Simple",
                    "predicted_price": float(forecast.iloc[-1]),
                    "forecast_24h": float(forecast.iloc[-1]),
                    "all_forecasts": forecast.tolist()
                }
            except:
                return {"model": "Exponential_Smoothing", "error": str(e)}
    
    # ENSEMBLE #

    def forecast_ensemble(self, data: pd.DataFrame, steps: int = 24) -> Dict:
        """
        Ensemble: Combines all models
        
        HOW IT WORKS:
        - Runs all 4 models
        - Takes weighted average
        - More robust than single model
        
        WEIGHTS:
        - ARIMA: 30% (good for short-term)
        - SARIMA: 20% (adds seasonality)
        - Prophet: 30% (robust to outliers)
        - ETS: 20% (smooths predictions)
        
        BEST FOR: Maximum accuracy
        
        Args:
            data: Historical price data
            steps: Hours to forecast
        
        Returns:
            Combined forecast from all models
        """
        try:
            logger.info("Running Ensemble forecast (all models)...")
            
            # Run all models
            arima_result = self.forecast_arima(data, steps)
            sarima_result = self.forecast_sarima(data, steps)
            prophet_result = self.forecast_prophet(data, steps)
            ets_result = self.forecast_exponential_smoothing(data, steps)
            
            # Collect predictions (handle errors)
            predictions = []
            weights = []
            
            if 'error' not in arima_result:
                predictions.append(arima_result['predicted_price'])
                weights.append(0.30)
            
            if 'error' not in sarima_result:
                predictions.append(sarima_result['predicted_price'])
                weights.append(0.20)
            
            if 'error' not in prophet_result:
                predictions.append(prophet_result['predicted_price'])
                weights.append(0.30)
            
            if 'error' not in ets_result:
                predictions.append(ets_result['predicted_price'])
                weights.append(0.20)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            
            # Weighted average
            ensemble_prediction = np.average(predictions, weights=weights)
            
            return {
                "model": "Ensemble",
                "predicted_price": float(ensemble_prediction),
                "forecast_24h": float(ensemble_prediction),
                "individual_predictions": {
                    "arima": arima_result.get('predicted_price'),
                    "sarima": sarima_result.get('predicted_price'),
                    "prophet": prophet_result.get('predicted_price'),
                    "exponential_smoothing": ets_result.get('predicted_price')
                },
                "weights_used": weights.tolist(),
                "models_succeeded": len(predictions)
            }
            
        except Exception as e:
            logger.error(f"Ensemble error: {e}")
            return {"model": "Ensemble", "error": str(e)}
    
    
    def forecast(
        self, 
        historical_data: List[Dict], 
        method: str = 'ensemble',
        steps: int = 24
    ) -> Dict:
        """
        Main forecasting method
        
        Args:
            historical_data: List of OHLCV data from Binance
            method: 'arima', 'sarima', 'prophet', 'ets', or 'ensemble'
            steps: Hours to forecast ahead
        
        Returns:
            Forecast results
        """
        # Prepare data
        data = self.prepare_data(historical_data)
        
        # Choose method
        if method == 'arima':
            return self.forecast_arima(data, steps)
        elif method == 'sarima':
            return self.forecast_sarima(data, steps)
        elif method == 'prophet':
            return self.forecast_prophet(data, steps)
        elif method == 'ets':
            return self.forecast_exponential_smoothing(data, steps)
        elif method == 'ensemble':
            return self.forecast_ensemble(data, steps)
        else:
            return {"error": f"Unknown method: {method}"}

price_forecaster = PriceForecaster()

