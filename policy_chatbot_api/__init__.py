"""
Policy Chatbot API Package

한국 지역 정책 검색을 위한 RESTful API 패키지
"""

__version__ = "1.0.8"
__author__ = "KDT Hackathon Team"
__email__ = "team@example.com"

from .policy_chatbot import PolicyChatbot
from .api_server import create_app

__all__ = ["PolicyChatbot", "create_app"] 