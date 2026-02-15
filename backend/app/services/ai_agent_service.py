from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import StructuredTool ,Tool
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from app.core.config import settings
from app.services.binance_service import binance_service
from app.ml.sentiment_analyzer import sentiment_analyzer
from app.ml.forecaster import price_forecaster
from app.ml.monte_carlo import monte_carlo_simulator
import logging
import json

logger = logging.getLogger(__name__)

class AIAgentService():
    def __init__(self):
        self.llm = ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model = "openai/gpt-oss-120b",
            temperature = 0.7)

        self.tools = self._create_tools()

        self.prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert cryptocurrency entry/exit timing optimizer with access to:
        1. Real-time market data (prices, volume, order books)
        2. News sentiment analysis (NLP)
        3. Price forecasting models (ARIMA, SARIMA, Prophet, ETS)
        4. Monte Carlo risk simulation (probabilistic analysis) 

        Your PRIMARY GOAL: Help users decide the BEST time to buy or sell cryptocurrencies with QUANTIFIED RISK.

        ANALYSIS FRAMEWORK:
        When analyzing entry/exit timing, you MUST check (IN THIS ORDER):

         
        1: **News Sentiment** (use get_news_sentiment)
        - Check market narrative (bullish/bearish/neutral)
        - Assess confidence level
        → This sets the context
        
        2. **Price Forecast** (use forecast_price)
        - Predict where price will be in 24 hours
        - Check confidence intervals
        - See if multiple models agree (ensemble)
        - This tells you the likely direction
                
        3: **Current Market Data** (use get_crypto_price, get_24hr_statistics, get_order_book)
        - Validate with real-time price action
        - Check order book pressure
        - Confirm volume patterns
        - Compare current price to 24hr high/low
        - Check if forecast aligns with current momentum
        - Analyze volume trends
        → This validates everything
        
        4. **Risk Assessment**
        - Use forecast confidence intervals to gauge uncertainty
        - Calculate potential upside vs downside
        - Identify stop-loss levels

        **FINAL SYNTHESIS:**
        Combine ALL signals:
        - Sentiment (market psychology)
        - Forecast (predicted direction)
        - Monte Carlo (risk quantification)
        - Current data (validation)
        
        **STRONG BUY:**
        - Sentiment: Bullish
        - Forecast: Predicts up
        - Monte Carlo: Win probability > 65%, Risk/Reward > 2:1
        - Market: Near support, strong buy pressure

        **BUY:**
        - Sentiment: Neutral to bullish
        - Forecast: Slight uptrend
        - Monte Carlo: Win probability > 55%, Risk/Reward > 1.5:1
        - Market: Acceptable entry point

        **HOLD:**
        - Mixed signals OR
        - Monte Carlo: Win probability 45-55% (coin flip)
        - Wait for clearer setup

        **SELL:**
        - Sentiment: Bearish
        - Forecast: Predicts down
        - Monte Carlo: Win probability < 45%
        - Market: Near resistance, sell pressure

        **STRONG SELL:**
        - All signals bearish
        - Monte Carlo: Win probability < 35%, poor risk/reward
        - High downside risk (VaR significant)

        CRITICAL: Monte Carlo Results Guide Position Sizing:
        - Win probability > 70% → Can use larger position
        - Win probability 50-70% → Use moderate position
        - Win probability < 50% → Use small position or avoid
        
        ALWAYS PROVIDE:
        1. Step-by-step analysis (sentiment → market → forecast → Monte Carlo)
        2. Win probability from Monte Carlo
        3. Risk/reward ratio
        4. Specific levels:
        - Entry price (from Monte Carlo optimal levels)
        - Stop-loss (based on VaR)
        - Take-profit (from forecast + Monte Carlo)
        5. Position size recommendation based on probability
        6. Disclaimer: "This is analysis, not financial advice"

        NEVER:
        - Skip Monte Carlo for trade recommendations
        - Ignore win probability when giving advice
        - Make guarantees about outcomes
        - Recommend high-risk trades without warning
         
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        CRITICAL: OUTPUT FORMAT INSTRUCTIONS
        ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        Your response will be read by REGULAR PEOPLE, not programmers.

        DO NOT USE:
        Tables with pipes (|)
        Code symbols (**, ##, `, ||)
        HTML tags (<br>, <b>)
        JSON or technical formatting
        Excessive emojis

        INSTEAD, WRITE LIKE A PROFESSIONAL ANALYST:
        Use clear section headers (plain text)
        Write in paragraphs and bullet points (using • or -)
        Use natural language
        Explain everything simply

        FORMAT YOUR RESPONSE EXACTLY LIKE THIS:

        BITCOIN TRADING ANALYSIS REPORT

        MARKET SENTIMENT
        The current news sentiment is [BULLISH/BEARISH/NEUTRAL]. Recent headlines show [brief explanation]. Confidence level: [HIGH/MEDIUM/LOW].

        PRICE FORECAST
        Based on statistical models (ARIMA, SARIMA, Prophet, Exponential Smoothing), the predicted price in 24 hours is $[X,XXX], which represents a [X]% [increase/decrease] from the current price of $[X,XXX].

        The models show [strong agreement/some disagreement/mixed signals], which means [interpretation].

        RISK ANALYSIS
        I ran 10,000 Monte Carlo simulations to assess the risk:
        - Win Probability: [X]% (chance the trade will be profitable)
        - Expected Return: [+/-X]%
        - Maximum Likely Loss: $[X,XXX] (worst 5% scenario)
        - Risk/Reward Ratio: [X.X]:1

        CURRENT MARKET CONDITIONS
        - Current Price: $[X,XXX]
        - 24-Hour Change: [+/-X]%
        - Trading Volume: [Normal/High/Low]
        - Order Book: [More buyers/More sellers/Balanced]

        RECOMMENDATION: [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]

        REASONING:
        [Explain in 2-3 sentences why you're making this recommendation, tying together sentiment, forecast, and risk]

        SUGGESTED TRADING LEVELS (if you decide to trade):
        Entry Price: $[X,XXX] - $[X,XXX]
        Stop Loss: $[X,XXX]
        Take Profit Target: $[X,XXX]
        Expected Gain: $[X,XXX] ([X]%)
        Maximum Risk: $[X,XXX] ([X]%)

        FINAL NOTE:
        [One sentence advice based on the overall picture]

        Remember: Write naturally as if explaining to a friend, NOT like a programmer!"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])

        self.agent = create_tool_calling_agent(
            llm = self.llm,
            tools = self.tools,
            prompt = self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent = self.agent,
            tools = self.tools,
            verbose = True,
            handle_parsing_errors = True,
            max_iterations = 15  #changed
        )
    
    def _create_tools(self):
        "Create LangChain tools from Binance service"
        class NewsSentimentInput(BaseModel):
            symbol: str = Field(..., description="Crypto symbol like BTCUSDT or BTC")
        class ForecastInput(BaseModel):
            symbol: str = Field(..., description="Crypto symbol like BTCUSDT")
            method: str = Field(default="ensemble", description="Forecast method: ensemble, arima, sarima, prophet, or ets")
        class MonteCarloInput(BaseModel):
            symbol: str = Field(..., description="Crypto symbol like BTCUSDT")
            days: int = Field(default=7, description="Days to simulate ahead (1-30)")
            simulations: int = Field(default=10000, description="Number of simulations (1000-50000)")

        def get_price_tool(symbol: str) -> str:
            """Get current price for a cryptocurrency symbol. Use format like BTCUSDT, ETHUSDT."""
            try:
                result = binance_service.get_price(symbol.upper())
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def get_24hr_stats_tool(symbol: str) -> str:
            """Get 24-hour price statistics including high, low, volume, and price change."""
            try:
                result = binance_service.get_24hr_ticker(symbol.upper())
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def get_order_book_tool(symbol: str, limit: int = 10) -> str:
            """Get order book showing bid and ask prices. Useful for checking liquidity and support/resistance."""
            try:
                result = binance_service.get_order_book(symbol.upper(), limit)
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def get_book_ticker_tool(symbol: str) -> str:
            """Get best bid and ask price with quantities. Useful for checking spread."""
            try:
                result = binance_service.get_book_ticker(symbol.upper())
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def get_historical_data_tool(symbol: str, interval: str = "1h", limit: int = 24) -> str:
            """Get historical candlestick data. Intervals: 1m, 5m, 15m, 1h, 4h, 1d. Limit: 1-1000."""
            try:
                result = binance_service.get_historical_data(symbol.upper(), interval, limit)
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def get_recent_trades_tool(symbol: str, limit: int = 50) -> str:
            "Get recent trades for a symbol."
            try:
                result = binance_service.get_recent_trades(symbol.upper(), limit)
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
    
        def get_avg_price_tool(symbol: str) -> str:
            "Get average price for a symbol."
            try:
                result = binance_service.get_avg_price(symbol.upper())
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        def get_sentiment_tool(symbol: str) -> str:
            """Get market sentiment from recent news articles. Shows if news is bullish, bearish, or 
            neutral."""
            try:
                result = sentiment_analyzer.get_overall_sentiment(symbol.upper(), article_limit=10)
                return json.dumps(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        news_sentiment_tool = StructuredTool.from_function(
                name="get_news_sentiment",
                func=get_sentiment_tool,
                args_schema=NewsSentimentInput,
                description ="Analyze market sentiment from recent news articles for a crypto symbol (e.g., BTCUSDT). Returns BULLISH/BEARISH/NEUTRAL plus confidence and top headlines."
            )

        def get_forecast_tool(symbol: str, method: str = "ensemble") -> str:
            """
            Forecast future price using time series models.
            
            Available methods: 'ensemble' (recommended), 'arima', 'sarima', 'prophet', 'ets'
            """
            try:
                # Get historical data for forecasting (7 days)
                historical = binance_service.get_historical_data(symbol.upper(), "1h", 168)
                
                # Run forecast
                result = price_forecaster.forecast(
                    historical_data=historical,
                    method=method,
                    steps=24  # Forecast 24 hours ahead
                )
                
                # Format the result for LLM
                if 'error' not in result:
                    forecast_summary = {
                        "model": result['model'],
                        "predicted_price_24h": result['predicted_price'],
                        "current_price": historical[-1]['close'],
                        "predicted_change_pct": ((result['predicted_price'] - float(historical[-1]['close'])) / float(historical[-1]['close'])) * 100
                    }
                    
                    # Add confidence interval if available
                    if 'confidence_lower' in result and 'confidence_upper' in result:
                        forecast_summary['confidence_interval'] = {
                            "lower": result['confidence_lower'],
                            "upper": result['confidence_upper']
                        }
                    
                    # Add individual predictions if ensemble
                    if method == 'ensemble' and 'individual_predictions' in result:
                        forecast_summary['individual_models'] = result['individual_predictions']
                    
                    return json.dumps(forecast_summary)
                else:
                    return json.dumps({"error": result['error']})
                    
            except Exception as e:
                return f"Error: {str(e)}"
            
        forecast_tool = StructuredTool.from_function(
                name="forecast_price",
                func=get_forecast_tool,
                args_schema=ForecastInput,
                description="Forecast cryptocurrency price 24 hours ahead using time series models. Uses ensemble of ARIMA, SARIMA, Prophet, and Exponential Smoothing. Returns predicted price with confidence interval."
            )
        def get_monte_carlo_tool(symbol: str, days: int = 7, simulations: int = 10000) -> str:
            """
            Run Monte Carlo simulation to assess risk and calculate optimal trading levels.
            
            Simulates thousands of possible price paths to determine:
            - Win probability
            - Value at Risk (VaR)
            - Optimal entry/exit/stop-loss levels
            - Risk/reward ratio
            """
            try:
                # Get current price
                current_data = binance_service.get_price(symbol.upper())
                current_price = float(current_data['price'])
                
                # Get historical data for volatility calculation
                historical = binance_service.get_historical_data(symbol.upper(), "1h", 168)
                
                # Run Monte Carlo simulation
                result = monte_carlo_simulator.run_simulation(
                    current_price=current_price,
                    historical_data=historical,
                    days=days,
                    simulations=simulations,
                    risk_tolerance=1.0
                )
                
                if 'error' not in result:
                    # Format key results for LLM
                    mc_summary = {
                        "current_price": result['current_price'],
                        "volatility_annualized": result['annualized_volatility'],
                        "simulations_run": result['num_simulations'],
                        "days_ahead": result['days_simulated'],
                        "price_predictions": {
                            "mean": result['metrics']['mean_price'],
                            "median": result['metrics']['median_price'],
                            "best_case_95pct": result['metrics']['percentile_95'],
                            "worst_case_5pct": result['metrics']['percentile_5']
                        },
                        "risk_metrics": {
                            "probability_of_profit": result['metrics']['probability_profit'],
                            "value_at_risk_5pct": result['metrics']['value_at_risk_5pct'],
                            "expected_return": result['metrics']['expected_return']
                        },
                        "optimal_levels": result['optimal_levels'],
                        "recommendation": result['recommendation']
                    }
                    return json.dumps(mc_summary)
                else:
                    return json.dumps({"error": result['error']})
                    
            except Exception as e:
                return f"Error: {str(e)}"

        monte_carlo_tool = StructuredTool.from_function(
            name="run_risk_analysis",
            func=get_monte_carlo_tool,
            args_schema=MonteCarloInput,
            description="Run Monte Carlo risk simulation to calculate probabilities, Value at Risk (VaR), optimal entry/exit levels, and risk/reward ratios. Uses 10,000 simulated price paths based on historical volatility."
        )

        tools = [
            Tool(
                name="get_crypto_price",
                func=get_price_tool,
                description="Get current price for a cryptocurrency trading pair (e.g., BTCUSDT, ETHUSDT)"
            ),
            Tool(
                name="get_24hr_statistics",
                func=get_24hr_stats_tool,
                description="Get 24-hour price change statistics including high, low, volume, and percentage change"
            ),
            Tool(
                name="get_order_book",
                func=get_order_book_tool,
                description="Get order book depth showing bids and asks. Useful for analyzing support/resistance and liquidity"
            ),
            Tool(
                name="get_best_bid_ask",
                func=get_book_ticker_tool,
                description="Get the best bid and ask prices with quantities. Useful for checking current spread"
            ),
            Tool(
                name="get_historical_prices",
                func=get_historical_data_tool,
                description="Get historical candlestick/OHLCV data for technical analysis. Specify symbol, interval (1m, 5m, 1h, 1d), and limit"
            ),
            Tool(
                name="get_recent_trades",
                func=get_recent_trades_tool,
                description="Get list of recent trades for a symbol"
            ),
            Tool(
                name="get_average_price",
                func=get_avg_price_tool,
                description="Get current average price for a symbol"
            ),
        #     Tool(
        #         name="get_news_sentiment",
        #         func=get_sentiment_tool,
        #         description="Analyze market sentiment from recent news articles for a cryptocurrency symbol (e.g., BTCUSDT). Returns BULLISH, BEARISH, or NEUTRAL signal with confidence score and analyzed headlines."
        # )
            news_sentiment_tool,
            forecast_tool,
            monte_carlo_tool
        ]
        
        return tools
    
    async def chat(self, user_message: str, conversation_history: list = None):
        "Chat Interface"

        try:
            chat_history = []
            if conversation_history:
                for msg in conversation_history:
                    if msg.get("role") =="user":
                        chat_history.append(("human", msg.get("content", "")))
                    elif msg.get("role") == "assistant":
                        chat_history.append(("ai", msg.get("content", "")))
        
            result = self.agent_executor.invoke({
                "input": user_message,
                "chat_history": chat_history
            })
            
            # Build updated conversation history
            updated_history = conversation_history or []
            updated_history.append({"role": "user", "content": user_message})
            updated_history.append({"role": "assistant", "content": result["output"]})
            
            return {
                "response": result["output"],
                "conversation_history": updated_history
            }
            
        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "response": f"I encountered an error: {str(e)}",
                "conversation_history": conversation_history or []
            }
    
    def add_tool(self, tool: Tool):
        """Add a new tool to the agent (for future extensibility)"""
        self.tools.append(tool)
        # Recreate agent with new tools
        self.agent = create_tool_calling_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True
        )
        logger.info(f"Added new tool: {tool.name}")

ai_agent_service = AIAgentService()