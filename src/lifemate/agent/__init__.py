"""Prediction interface; evaluator labels are never part of AgentInput."""
from .base import Agent, AgentInput, Finish, ToolDecision
__all__ = ['Agent', 'AgentInput', 'Finish', 'ToolDecision']
