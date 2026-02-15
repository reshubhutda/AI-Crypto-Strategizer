# from fastapi import APIRouter, HTTPException
# from pydantic import BaseModel
# from typing import List, Optional, Dict, Any
# from app.services.ai_agent_service import ai_agent_service
# from app.services.response_formatter import response_formatter


# router = APIRouter()

# class ChatRequest(BaseModel):
#     message: str
#     conversation_history: Optional[List[dict]] = None

# class ChatResponse(BaseModel):
#     status: str
#     recommendation: str
#     analysis: str
#     metadata: Dict[str, Any]

# @router.post("/chat", response_model=ChatResponse)
# async def chat_with_agent(request: ChatRequest):
#     """
#     Chat with AI agent about cryptocurrency markets
    
#     Returns clean, structured JSON response
#     """
#     try:
#         # Get response from agent
#         result = await ai_agent_service.chat(
#             user_message=request.message,
#             conversation_history=request.conversation_history
#         )
        
#         # Format into clean response
#         formatted = {
#             "status": "success",
#             "recommendation": response_formatter._extract_recommendation(result['response']),
#             "analysis": result['response'],
#             "metadata": {
#                 "conversation_turns": len(result.get('conversation_history', [])) // 2,
#                 "tools_used": "sentiment, forecast, monte_carlo, market_data"
#             }
#         }
        
#         return formatted
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @router.get("/tools")
# async def list_tools():
#     """List available tools the agent can use"""
#     tools_info = [
#         {
#             "name": tool.name,
#             "description": tool.description
#         }
#         for tool in ai_agent_service.tools
#     ]
#     return {"tools": tools_info}
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from app.services.ai_agent_service import ai_agent_service
from app.services.report_generator import report_generator

router = APIRouter()

class ChatRequest(BaseModel):
    message: str = Field(
        ..., 
        description="Your question about cryptocurrency trading",
        example="Should I buy Bitcoin now? Run complete analysis."
    )
    conversation_history: Optional[List[dict]] = Field(
        default=None,
        description="Previous conversation (optional)"
    )

class AnalysisResult(BaseModel):
    recommendation: str = Field(description="STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL")
    analysis: str = Field(description="Complete analysis from AI agent")
    confidence: str = Field(description="Confidence level of recommendation")

class ChatResponse(BaseModel):
    status: str = Field(default="success")
    result: AnalysisResult
    metadata: Dict[str, Any] = Field(description="Additional information")

    class Config:
        schema_extra = {
            "example": {
                "status": "success",
                "result": {
                    "recommendation": "HOLD",
                    "analysis": "Based on sentiment analysis (BULLISH), price forecast (slight decline to $67k), and Monte Carlo simulation (49% win probability), the recommendation is to HOLD. Entry: $66,200, Stop: $67,200, Target: $68,300",
                    "confidence": "MEDIUM"
                },
                "metadata": {
                    "tools_used": ["sentiment", "forecast", "monte_carlo"],
                    "timestamp": "2026-02-15T10:30:00Z"
                }
            }
        }

@router.post("/chat", 
             response_model=ChatResponse,
             summary="Get Trading Recommendation",
             description="""
Analyze cryptocurrency market and get AI-powered trading recommendations.

**The AI will:**
1. Check news sentiment (bullish/bearish)
2. Forecast price using time series models
3. Run Monte Carlo risk simulation (10,000 scenarios)
4. Provide specific entry/exit/stop-loss levels
5. Calculate win probability and risk/reward ratio

**Example request:**
```json
{
  "message": "Should I buy Bitcoin now? Run complete analysis."
}
```
""")
async def chat_with_agent(request: ChatRequest):
    """Get AI-powered trading analysis and recommendations"""
    try:
        result = await ai_agent_service.chat(
            user_message=request.message,
            conversation_history=request.conversation_history
        )
        
        # Clean up the response
        analysis_text = result['response']
        
        # Extract recommendation
        rec = "NEUTRAL"
        if "STRONG BUY" in analysis_text.upper():
            rec = "STRONG_BUY"
        elif "BUY" in analysis_text.upper():
            rec = "BUY"
        elif "HOLD" in analysis_text.upper():
            rec = "HOLD"
        elif "SELL" in analysis_text.upper():
            rec = "SELL"
        
        # Extract confidence if mentioned
        confidence = "MEDIUM"
        if "high confidence" in analysis_text.lower() or "probability > 70" in analysis_text.lower():
            confidence = "HIGH"
        elif "low confidence" in analysis_text.lower() or "probability < 45" in analysis_text.lower():
            confidence = "LOW"
        
        return {
            "status": "success",
            "result": {
                "recommendation": rec,
                "analysis": analysis_text,
                "confidence": confidence
            },
            "metadata": {
                "tools_used": ["sentiment_analysis", "price_forecast", "monte_carlo_simulation", "market_data"],
                "conversation_turns": len(result.get('conversation_history', [])) // 2
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze", 
             summary="Get Analysis Report (Human-Readable)",
             description="Get a clean, human-readable trading analysis report")
async def get_analysis_report(request: ChatRequest):
    """
    Get AI analysis as a clean, readable report (NOT JSON)
    
    Perfect for sharing with non-technical users or printing
    """
    try:
        # Get analysis from agent
        result = await ai_agent_service.chat(
            user_message=request.message,
            conversation_history=request.conversation_history
        )
        
        # Generate clean report
        report = report_generator.generate_report(result['response'])
        
        # Return as plain text (not JSON!)
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content=report, media_type="text/plain")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
