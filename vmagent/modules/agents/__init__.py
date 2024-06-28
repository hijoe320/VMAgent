from .ACAgent import ACAgent
from .DQNAgent_modify import DQNAgent_modify
from .DQNAgent import DQNAgent

REGISTRY = {}

REGISTRY['ACAgent'] = ACAgent
REGISTRY['DQNAgent_modify'] = DQNAgent_modify
REGISTRY['DQNAgent'] = DQNAgent