from fastapi import APIRouter, HTTPException, Query
from app.services.binance_service import binance_service

router = APIRouter()

@router.get("/price/{symbol}")
async def get_price(symbol:str):
    try:
        result = binance_service.get_price(symbol)
        return result
    except Exception as e:
        raise HTTPException(status_code = 400, detail = str(e))
    
@router.get("/historical/{symbol}")
async def get_historical(
    symbol:str, interval:str = Query("1h", regex="^(1m|5m|15m|1h|4h|1d)$"), 
    limit:int =Query(100, ge=1, le=1000)
):
    try:
        result = binance_service.get_historical_data(symbol, interval, limit)
        return {
            "symbol": symbol,
            "interval": interval,
            "data": result
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/exchange-info")
async def get_exchange_info():
    try:
        return binance_service.get_exchange_info()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/order-book/{symbol}")
async def get_order_book(
    symbol: str,
    limit: int = Query(10, ge=5, le=100)
):
    try:
        return binance_service.get_order_book(symbol, limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/ticker-24hr/{symbol}")
async def get_24hr_ticker(symbol: str):
    try:
        return binance_service.get_24hr_ticker(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/recent-trades/{symbol}")
async def get_recent_trades(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000)
):
    try:
        return binance_service.get_recent_trades(symbol, limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/all-tickers")
async def get_all_tickers():
    try:
        return binance_service.get_all_tickers()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/avg-price/{symbol}")
async def get_avg_price(symbol: str):
    try:
        return binance_service.get_avg_price(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/ping")
async def ping():
    "Test connectivity to Binance"
    try:
        return binance_service.ping()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/server-time")
async def get_server_time():
    try:
        return binance_service.get_server_time()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/book-ticker/{symbol}")
async def get_book_ticker(symbol: str):
    "Get best bid/ask price and quantity"
    try:
        return binance_service.get_book_ticker(symbol)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/aggregate-trades/{symbol}")
async def get_aggregate_trades(
    symbol: str,
    limit: int = Query(100, ge=1, le=1000)
):
    "Get aggregate trades"
    try:
        return binance_service.get_aggregate_trades(symbol, limit)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/all-book-tickers")
async def get_all_book_tickers():
    "Get best bid/ask for all symbols"
    try:
        return binance_service.get_all_book_tickers()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "binance-api"}

