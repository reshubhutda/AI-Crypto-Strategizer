from typing import Dict
import re
from datetime import datetime

class ReportGenerator:
    """Generate professional, human-readable reports"""
    
    @staticmethod
    def generate_report(agent_response: str) -> str:
        """
        Convert AI agent response into a clean business report
        
        NO technical symbols, NO JSON, NO code formatting
        Just plain English that anyone can read
        """
        
        # Clean up technical symbols
        clean_text = agent_response
        
        # Remove markdown symbols
        clean_text = re.sub(r'\*\*', '', clean_text)
        clean_text = re.sub(r'#{1,}', '', clean_text)
        clean_text = re.sub(r'\|', '', clean_text)  # Remove table pipes
        clean_text = re.sub(r'<br>', '\n', clean_text)  # Convert <br> to newlines
        clean_text = re.sub(r'`', '', clean_text)  # Remove backticks
        clean_text = re.sub(r'-{2,}', '', clean_text)  # Remove dashes
        
        # Remove excessive emojis (keep just a few for visual breaks)
        # But make it more professional
        
        # Get current date/time
        timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        
        # Build professional report
        report = f"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   CRYPTOCURRENCY INVESTMENT ANALYSIS                       ║
║                         Generated: {timestamp:<40} ║
╚════════════════════════════════════════════════════════════════════════════╝

{clean_text}

────────────────────────────────────────────────────────────────────────────

IMPORTANT DISCLAIMER

This analysis is provided for informational and educational purposes only. 
It should NOT be considered as financial, investment, trading, or any other 
type of advice.

Cryptocurrency markets are highly volatile and speculative. You should:
  • Conduct your own research and due diligence
  • Consult with a qualified financial advisor before making investment decisions
  • Only invest money you can afford to lose
  • Understand that past performance does not guarantee future results

The analysis provided uses mathematical models and historical data, but cannot 
predict future market movements with certainty. All investments carry risk.

────────────────────────────────────────────────────────────────────────────
                              END OF REPORT
════════════════════════════════════════════════════════════════════════════
"""
        
        return report
    
    @staticmethod
    def _extract_key_info(text: str) -> Dict[str, str]:
        """Extract key information for structured summary"""
        
        # Try to find recommendation
        recommendation = "NOT FOUND"
        if "STRONG BUY" in text.upper():
            recommendation = "STRONG BUY"
        elif "BUY" in text.upper():
            recommendation = "BUY"
        elif "HOLD" in text.upper():
            recommendation = "HOLD"
        elif "SELL" in text.upper():
            recommendation = "SELL"
        
        return {
            "recommendation": recommendation
        }

report_generator = ReportGenerator()