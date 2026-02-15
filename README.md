# AI-Powered Cryptocurrency Trading Optimizer

An intelligent system that combines NLP sentiment analysis, statistical forecasting, and Monte Carlo simulation to provide data-driven cryptocurrency trading recommendations through a FastAPI backend.

## Project Overview

This project answers a simple question: **"Should I buy this cryptocurrency right now?"**

Instead of relying on gut feeling or single indicators, the system analyzes the market through three different lenses:

**Sentiment Analysis** - Fetches recent crypto news and uses FinBERT (BERT fine-tuned for financial text) to determine if the market mood is bullish, bearish, or neutral.
**Price Forecasting** - Uses four statistical models (ARIMA, SARIMA, Prophet, Exponential Smoothing) in parallel and combines them into an ensemble forecast for 24-hour price prediction.
**Risk Simulation** - Runs 10,000 Monte Carlo simulations using Geometric Brownian Motion to calculate win probability, Value at Risk, and optimal entry/exit levels.

An AI agent powered by LangChain and Groq orchestrates all three analyses, synthesizes the results, and generates a human-readable trading report.

![Image](https://github.com/reshubhutda/AI-Crypto-Strategizer/blob/main/AI%20Agent%20Flow.png)

## Data Pipeline: Binance Integration

**Real-Time Market Data**

The system integrates directly with Binance.US API to fetch live cryptocurrency data. No API keys are required for public market endpoints, making it accessible for anyone to use.

**Market Data Endpoints Created:**
- Current price for any trading pair (BTCUSDT, ETHUSDT, etc.)
- 24-hour statistics (high, low, volume, price change percentage)
- Historical OHLCV data (Open, High, Low, Close, Volume candlesticks)
- Order book depth (bids and asks for liquidity analysis)
- Exchange trading rules and available symbols

**Binance Service Wrapper**
Built a clean service layer (`binance_service.py`) that wraps the `python-binance` SDK. This handles:
- Error handling and retries for API failures
- Data normalization (converting timestamps, formatting prices)
- Rate limiting to avoid hitting Binance restrictions
- Consistent response formats across all endpoints

The wrapper exposes simple Python functions like `get_price(symbol)`, `get_historical_data(symbol, interval, limit)`, and `get_24hr_ticker(symbol)` that return clean JSON.

## Building the AI Agent Architecture

**Component 1: Sentiment Analysis Module**

**Data Source:** NewsData.io API fetches the 10 most recent cryptocurrency news articles.

**Processing:** Each headline is analyzed using FinBERT, a BERT model specifically trained on financial text. Unlike generic sentiment models, FinBERT understands financial terminology like "bearish outlook" or "bullish momentum."

**Output:** Returns sentiment classification (BULLISH/BEARISH/NEUTRAL) with confidence scores ranging from -1 (extremely bearish) to +1 (extremely bullish).

**Implementation:** Located in `ml/sentiment_analyzer.py`, uses HuggingFace transformers library with the ProsusAI/finbert model.

**Component 2: Price Forecasting Module**

**Data Source:** 168 hours (7 days) of hourly OHLCV data from Binance.

**Four Forecasting Models:**
1. **ARIMA** - Auto-Regressive Integrated Moving Average analyzes how past prices influence future prices. Uses `pmdarima` for automatic parameter selection.
2. **SARIMA** - Seasonal ARIMA captures repeating patterns (like weekend effects in crypto markets) with a 24-hour seasonal period.
3. **Prophet** - Facebook's forecasting library decomposes time series into trend, seasonality, and outliers. Particularly robust to missing data.
4. **Exponential Smoothing** - Gives more weight to recent prices, making it responsive to sudden market changes.

**Ensemble Approach:** All four models run independently, producing their own forecasts. These are combined using weighted averaging (ARIMA 30%, Prophet 30%, SARIMA 20%, ETS 20%) for a more robust prediction than any single model.

**Output:** 24-hour price forecast with confidence intervals showing the range of possible outcomes.

**Implementation:** Located in `ml/forecaster.py`, uses statsmodels, prophet, and pmdarima libraries.


**Component 3: Monte Carlo Risk Simulator**

**Data Source:** Same 168 hours of price data to calculate historical volatility.

**Simulation Process:**

1. Calculate volatility (standard deviation of returns) from historical data
2. Use Geometric Brownian Motion formula: `S(t+1) = S(t) * exp((μ - σ²/2)dt + σ√dt*Z)`
3. Run 10,000 simulated price paths, each spanning 7 days into the future
4. Each path includes random shocks based on historical volatility

**Statistical Analysis:**

From the 10,000 simulation outcomes, the system calculates:
- **Win Probability:** Percentage of simulations that end above current price
- **Value at Risk (VaR):** Maximum loss in the worst 5% of scenarios
- **Percentiles:** Best case (95th), likely range (25th-75th), worst case (5th)
- **Optimal Levels:** Entry price, exit target, and stop-loss based on simulation distribution

**Output:** Risk metrics and optimal trading levels with mathematical backing from thousands of scenarios.

**Implementation:** Located in `ml/monte_carlo.py`, uses NumPy for vectorized calculations.


## System Integration: LangChain Agent

**Agent Orchestration**

Rather than hardcoding the analysis workflow, LangChain's AgentExecutor lets the AI decide which tools to use based on the user's question.

**Tool Definition:**

Each analysis component (sentiment, forecast, monte carlo) is wrapped as a LangChain StructuredTool with:
- Name and description (so the LLM knows what it does)
- Input schema using Pydantic (type-safe parameters)
- Execution function that calls the underlying ML module

**Agent Decision Flow:**

1. User asks: "Should I buy Bitcoin now?"
2. LLM (Llama 3.1 70B via Groq) understands the intent
3. Agent decides to call: sentiment tool → forecast tool → monte carlo tool → market data
4. Each tool returns results back to the agent
5. LLM synthesizes all data and generates recommendation

**System Prompt Engineering:**

The agent's behavior is guided by a carefully crafted system prompt that:
- Defines the analysis workflow (sentiment → forecast → risk → market validation)
- Sets decision criteria (when to recommend BUY vs HOLD vs SELL)
- Specifies output format (clear sections, specific price levels, plain English)

**Implementation:** Located in `services/ai_agent_service.py`, uses langchain and langchain-groq.


## FastAPI Backend Implementation

**API Structure**

The backend is split into two route modules:

**Market Data Routes** (`api/routes.py`):
- Direct access to Binance data without AI processing
- Useful for building dashboards or other applications
- Endpoints: `/price/{symbol}`, `/historical/{symbol}`, `/ticker-24hr/{symbol}`, `/order-book/{symbol}`

**AI Agent Routes** (`api/agent_routes.py`):
- Intelligent analysis powered by the LangChain agent
- Two variants: `/chat` (JSON response) and `/analyze` (human-readable report)
- Handles conversation history for follow-up questions

**Request Processing Flow:**

1. User sends POST request to `/api/v1/agent/analyze` with natural language query
2. FastAPI validates request body using Pydantic models
3. Request is passed to AI agent service
4. Agent orchestrates analysis using all three ML modules
5. Raw analysis is formatted by report generator
6. Clean report returned as plain text response

**Auto-Generated Documentation:**

FastAPI automatically generates interactive Swagger UI at `/docs`. Users can test all endpoints directly from the browser without writing any code.

## Report Generation and Output Formatting

**Human-Readable Reports**

The `report_generator.py` service converts technical AI output into professional business reports.

**Formatting Process:**

1. Remove technical symbols (markdown, code formatting, pipes)
2. Replace HTML tags with proper line breaks
3. Structure content into clear sections with headers
4. Add professional header with timestamp
5. Include comprehensive disclaimer


**Report Sections:**

- **Market Sentiment:** Explains news analysis and market mood
- **Price Forecast:** Shows model predictions and consensus
- **Risk Analysis:** Presents Monte Carlo results with probabilities
- **Current Market Conditions:** Validates with real-time data
- **Recommendation:** Clear BUY/SELL/HOLD decision with reasoning
- **Trading Levels:** Specific entry, stop-loss, and take-profit prices

**Example Output:**
```
╔════════════════════════════════════════════════════════════════╗
║           CRYPTOCURRENCY INVESTMENT ANALYSIS                   ║
╚════════════════════════════════════════════════════════════════╝

BITCOIN TRADING ANALYSIS REPORT

MARKET SENTIMENT
The current news sentiment is BULLISH with moderate confidence...

PRICE FORECAST
Based on ensemble forecasting, predicted 24h price: $67,002...

RISK ANALYSIS
Monte Carlo simulation (10,000 paths) shows:
- Win Probability: 49.8%
- Value at Risk: $2,561
...

RECOMMENDATION: HOLD
```

---

## Project File Structure
```
backend/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py                 # Binance market data endpoints
│   │   └── agent_routes.py           # AI analysis endpoints (/chat, /analyze)
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   └── config.py                 # Environment config with pydantic-settings
│   │
│   ├── ml/                           # Machine Learning modules
│   │   ├── __init__.py
│   │   ├── forecaster.py             # ARIMA, SARIMA, Prophet, ETS models
│   │   ├── monte_carlo.py            # GBM simulation with NumPy
│   │   └── sentiment_analyzer.py     # FinBERT NLP analysis
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── ai_agent_service.py       # LangChain agent orchestration
│   │   ├── binance_service.py        # Binance API wrapper
│   │   └── report_generator.py       # Output formatting
│   │
│   └── main.py                       # FastAPI app entry point
│
├── tests/
│   ├── test_forecaster.py
│   ├── test_monte_carlo.py
│   └── test_sentiment.py
│
├── .env                              # API keys (not committed)
├── .gitignore
├── requirements.txt                  # Python dependencies
└── README.md
```

**Key Files Explained:**

`main.py` - FastAPI application initialization, includes both routers, serves Swagger UI

`binance_service.py` - Wrapper around python-binance SDK with error handling and data normalization

`sentiment_analyzer.py` - Fetches news from NewsData.io, processes with FinBERT model

`forecaster.py` - Implements all four forecasting models and ensemble averaging

`monte_carlo.py` - Geometric Brownian Motion simulation with statistical analysis

`ai_agent_service.py` - LangChain agent with tool definitions and system prompt

`report_generator.py` - Formats AI output into clean business reports

`config.py` - Loads environment variables using pydantic-settings for type safety

---

## Installation and Setup

**Prerequisites:**
- Python 3.10+
- Groq API key (free at console.groq.com)
- NewsData.io API key (free tier: 200 credits)

**Steps:**
```bash
# Clone and navigate
git clone <your-repo-url>
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
# Create .env file with:
GROQ_API_KEY=your_key_here
NEWSDATA_API_KEY=your_key_here

# Run server
uvicorn app.main:app --reload

# Access API docs
# Open browser: http://127.0.0.1:8000/docs
```

---

## Key Technologies

**Backend Framework:** FastAPI - Modern Python web framework with automatic OpenAPI docs
**AI Orchestration:** LangChain - Tool calling and agent workflow management
**LLM:** Groq (OpenAI/ gpt 120-b) - Fast inference for analysis synthesis
**NLP:** HuggingFace Transformers - FinBERT sentiment model
**Forecasting:** statsmodels (ARIMA/SARIMA), Prophet, pmdarima
**Simulation:** NumPy - Vectorized Monte Carlo calculations
**Data Processing:** Pandas - Time series manipulation
**Market Data:** python-binance - Binance API client
**News Data:** NewsData.io - Cryptocurrency news articles


## Usage Example

**Request:**
```bash
POST /api/v1/agent/analyze
{
  "message": "Should I buy Bitcoin now? Give me complete analysis."
}
```

**Response:** Professional report with sentiment, forecast, risk analysis, and specific trading levels
**Processing Time:** 15-20 seconds for complete analysis
