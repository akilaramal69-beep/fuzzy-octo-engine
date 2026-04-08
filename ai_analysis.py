#!/usr/bin/env python3
"""
AI Analysis Module - Deep Learning Token & Trade Guardian
Interfaces with Groq Cloud (LPU) for high-speed LLM analysis
"""

import logging
import json
from typing import Dict, Any, Optional
import aiohttp
from config import config

logger = logging.getLogger(__name__)

class AIAnalyzer:
    def __init__(self):
        self.api_key = config.GROQ_API_KEY
        self.model = config.GROQ_MODEL
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"

    async def analyze_launch(self, token_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Performs deep behavioral analysis on a new token launch.
        Args:
            token_data: Includes mint, creator history, dev holding, and launch tx data.
        """
        if not self.api_key:
            return {"confidence": 50, "reasoning": "AI Analysis skipped (No API Key)"}

        prompt = f"""
        Act as a professional Solana chain analyst specializing in Pump.fun rug-pull detection.
        Analyze this new token launch data and provide a confidence score (0-100) for trading.
        
        TOKEN DATA:
        - Mint: {token_data.get('mint')}
        - Creator: {token_data.get('creator')}
        - Dev Holding: {token_data.get('dev_holding_pct', 0):.2%}
        - Dev Activity: {token_data.get('coins_per_hour', 0)} launches in the last hour
        - Initial Buy: {token_data.get('initial_buy_pct', 0):.2%} of total supply
        
        Identify subtle patterns: Is this a "Launch Farm"? Does the dev holding look dangerous?
        Respond in JSON format: {{"confidence_score": integer, "is_rug_likely": boolean, "top_risk": "string", "reasoning": "string"}}
        """

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.2,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=5 # Sniper-speed timeout
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result['choices'][0]['message']['content']
                        return json.loads(content)
                    else:
                        logger.error(f"AI API Error: {resp.status}")
                        return {"confidence": 50, "reasoning": "Fallback to base algorithm"}
        except Exception as e:
            logger.error(f"AI Analysis failed: {e}")
            return {"confidence": 50, "reasoning": "Local scoring primary"}

    async def analyze_exit_timing(self, trade_data: Dict[str, Any], price_history: list) -> Dict[str, Any]:
        """
        Analyzes price momentum and volume to find the 'Best Moment' to sell.
        """
        if not self.api_key:
            return {"signal": "HOLD"}

        prompt = f"""
        Analyze this active sniper trade on Solana. Find the mathematically optimal exit point.
        
        TRADE DATA:
        - Entry Price: {trade_data.get('entry_price'):.9f}
        - Current Price: {trade_data.get('current_price'):.9f}
        - PnL: {trade_data.get('pnl_pct', 0):+.1f}%
        - Peak Reached: {trade_data.get('trailing_high'):.9f}
        
        PRICE TREND (Last 5 mins):
        {price_history[-10:] if price_history else "N/A"}
        
        Is this a 'Dead Cat Bounce' or a 'Boll Flag'? Should we sell now to lock in profit?
        Respond in JSON format: {{"signal": "SELL" | "HOLD", "confidence": 0-100, "reason": "string"}}
        """

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.base_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "response_format": {"type": "json_object"}
                    },
                    timeout=3
                ) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        content = result['choices'][0]['message']['content']
                        return json.loads(content)
        except Exception:
            pass
        return {"signal": "HOLD"}
