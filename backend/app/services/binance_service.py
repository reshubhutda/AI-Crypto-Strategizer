from binance.client import Client
from binance.exceptions import BinanceAPIException
from app.core.config import settings
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class BinanceService:
    def __init__(self):
        # For now, public data only (no API keys needed)
        self.client = Client(
            api_key="",
            api_secret="",
            tld='us'  # This makes it use binance.us!
        )
    
    def get_price(self, symbol: str) -> Dict:
        """Get current price for a symbol"""
        try:
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            return {
                "symbol": ticker['symbol'],
                "price": float(ticker['price'])
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching price: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_historical_data(self, symbol: str, interval: str = '1h', limit: int = 100) -> List[Dict]:
        """Get historical kline/candlestick data"""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            # Format the data nicely
            formatted_data = []
            for kline in klines:
                formatted_data.append({
                    "timestamp": kline[0],
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5])
                })
            
            return formatted_data
        
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching historical data: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
        
    def get_exchange_info(self) -> Dict:
        try:
            info = self.client.get_exchange_info()
            return {
                "timezone": info['timezone'],
                "serverTime": info['serverTime'],
                "symbols": [
                    {
                        "symbol": s['symbol'],
                        "status": s['status'],
                        "baseAsset": s['baseAsset'],
                        "quoteAsset": s['quoteAsset']
                    }
                    for s in info['symbols'][:50]  # Limit to first 50
                ]
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching exchange info: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_order_book(self, symbol: str, limit: int = 10) -> Dict:
        try:
            depth = self.client.get_order_book(symbol=symbol, limit=limit)
            return {
                "symbol": symbol,
                "bids": [[float(price), float(qty)] for price, qty in depth['bids']],
                "asks": [[float(price), float(qty)] for price, qty in depth['asks']]
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching order book: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_24hr_ticker(self, symbol: str) -> Dict:
        try:
            ticker = self.client.get_ticker(symbol=symbol)
            return {
                "symbol": ticker['symbol'],
                "priceChange": float(ticker['priceChange']),
                "priceChangePercent": float(ticker['priceChangePercent']),
                "lastPrice": float(ticker['lastPrice']),
                "highPrice": float(ticker['highPrice']),
                "lowPrice": float(ticker['lowPrice']),
                "volume": float(ticker['volume']),
                "quoteVolume": float(ticker['quoteVolume'])
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching 24hr ticker: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        try:
            trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            return [
                {
                    "id": trade['id'],
                    "price": float(trade['price']),
                    "qty": float(trade['qty']),
                    "time": trade['time'],
                    "isBuyerMaker": trade['isBuyerMaker']
                }
                for trade in trades
            ]
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching recent trades: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_all_tickers(self) -> List[Dict]:
        try:
            tickers = self.client.get_all_tickers()
            return [
                {
                    "symbol": ticker['symbol'],
                    "price": float(ticker['price'])
                }
                for ticker in tickers
            ]
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching all tickers: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_avg_price(self, symbol: str) -> Dict:
        try:
            avg = self.client.get_avg_price(symbol=symbol)
            return {
                "symbol": symbol,
                "price": float(avg['price']),
                "mins": avg['mins']
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching average price: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")

    def ping(self) -> Dict:
        "Test connectivity to Binance API"
        try:
            result = self.client.ping()
            return {"status": "connected", "ping": "success"}
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error pinging Binance: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")
    
    def get_server_time(self) -> Dict:
        try:
            result = self.client.get_server_time()
            return {
                "serverTime": result['serverTime'],
                "serverTimeReadable": result['serverTime']
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching server time: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")


    def get_book_ticker(self, symbol: str) -> Dict:
        """Get best bid/ask price and quantity"""
        try:
            ticker = self.client.get_orderbook_ticker(symbol=symbol)
            return {
                "symbol": ticker['symbol'],
                "bidPrice": float(ticker['bidPrice']),
                "bidQty": float(ticker['bidQty']),
                "askPrice": float(ticker['askPrice']),
                "askQty": float(ticker['askQty'])
            }
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching book ticker: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")

    def get_aggregate_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        try:
            trades = self.client.get_aggregate_trades(symbol=symbol, limit=limit)
            return [
                {
                    "aggTradeId": trade['a'],
                    "price": float(trade['p']),
                    "quantity": float(trade['q']),
                    "firstTradeId": trade['f'],
                    "lastTradeId": trade['l'],
                    "timestamp": trade['T'],
                    "isBuyerMaker": trade['m']
                }
                for trade in trades
            ]
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching aggregate trades: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")

    def get_all_book_tickers(self) -> List[Dict]:
        "Get best bid/ask for all symbols"
        try:
            tickers = self.client.get_orderbook_tickers()
            return [
                {
                    "symbol": ticker['symbol'],
                    "bidPrice": float(ticker['bidPrice']),
                    "bidQty": float(ticker['bidQty']),
                    "askPrice": float(ticker['askPrice']),
                    "askQty": float(ticker['askQty'])
                }
                for ticker in tickers[:50]  # Limit to 50 for performance
            ]
        except BinanceAPIException as e:
            logger.error(f"Binance API error: {e}")
            raise Exception(f"Error fetching all book tickers: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise Exception(f"Unexpected error: {str(e)}")

# Create a singleton instance
binance_service = BinanceService()