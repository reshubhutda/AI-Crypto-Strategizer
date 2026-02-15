from typing import Dict
import json

class ResponseFormatter:
    """Format AI agent responses into clean, professional JSON"""
    
    @staticmethod
    def format_analysis(agent_response: str, conversation_history: list) -> Dict:
        """
        Convert agent's text response into structured JSON
        
        Args:
            agent_response: Raw text from AI agent
            conversation_history: Conversation history
            
        Returns:
            Clean, structured response
        """
        
        # Extract key information from response
        # (We'll parse the agent's structured thinking)
        
        return {
            "status": "success",
            "analysis": {
                "summary": agent_response[:200] + "..." if len(agent_response) > 200 else agent_response,
                "full_analysis": agent_response,
                "timestamp": None  # Add if needed
            },
            "recommendation": ResponseFormatter._extract_recommendation(agent_response),
            "conversation_id": len(conversation_history) // 2  # Simple ID based on turns
        }
    
    @staticmethod
    def _extract_recommendation(text: str) -> str:
        """Extract the main recommendation from response"""
        # Look for key decision words
        text_upper = text.upper()
        
        if "STRONG BUY" in text_upper:
            return "STRONG_BUY"
        elif "BUY" in text_upper and "NOT" not in text_upper:
            return "BUY"
        elif "HOLD" in text_upper:
            return "HOLD"
        elif "STRONG SELL" in text_upper or "AVOID" in text_upper:
            return "STRONG_SELL"
        elif "SELL" in text_upper:
            return "SELL"
        else:
            return "NEUTRAL"

response_formatter = ResponseFormatter()