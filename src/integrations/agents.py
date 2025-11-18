"""
Kapittel 9: AI Agents
Implementation of simple and multi-agent systems.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from anthropic import Anthropic
from enum import Enum

try:
    from utils import config, logger, LoggerMixin
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from utils import config, logger, LoggerMixin


@dataclass
class AgentMessage:
    """Message between agents"""
    sender: str
    recipient: str
    content: str
    message_type: str = "info"  # info, task, result, error


class SimpleAgent(LoggerMixin):
    """
    Simple AI agent with tool calling capabilities.
    """
    
    def __init__(
        self,
        name: str,
        system_prompt: str,
        model: Optional[str] = None
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.model = model or config.ai.model
        self.client = Anthropic(api_key=config.ai.api_key)
        self.conversation_history: List[Dict[str, str]] = []
        self.tools: Dict[str, Callable] = {}
        
        self.log_info(f"Initialized agent: {name}")
    
    def register_tool(self, name: str, function: Callable, description: str):
        """Register a tool the agent can use."""
        self.tools[name] = {"function": function, "description": description}
        self.log_info(f"Registered tool: {name}")
    
    def run(self, task: str) -> str:
        """
        Execute task with potential tool calls.
        
        Args:
            task: Task description
            
        Returns:
            Agent's response
        """
        self.log_info(f"Agent {self.name} executing: {task}")
        
        # Add task to history
        self.conversation_history.append({
            "role": "user",
            "content": task
        })
        
        # Create messages with system prompt
        messages = [
            {"role": "user", "content": self.system_prompt},
            {"role": "assistant", "content": "I understand my role and am ready to help."}
        ] + self.conversation_history
        
        # Call AI
        response = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            messages=messages
        )
        
        answer = response.content[0].text
        
        # Add to history
        self.conversation_history.append({
            "role": "assistant",
            "content": answer
        })
        
        return answer


class MultiAgentSystem(LoggerMixin):
    """
    System for coordinating multiple AI agents.
    """
    
    def __init__(self):
        self.agents: Dict[str, SimpleAgent] = {}
        self.message_queue: List[AgentMessage] = []
        self.log_info("Initialized multi-agent system")
    
    def add_agent(self, agent: SimpleAgent):
        """Add agent to system."""
        self.agents[agent.name] = agent
        self.log_info(f"Added agent: {agent.name}")
    
    def send_message(self, message: AgentMessage):
        """Send message between agents."""
        self.message_queue.append(message)
        self.log_info(f"Message: {message.sender} -> {message.recipient}")
    
    def execute_workflow(
        self,
        task: str,
        workflow: List[str]
    ) -> Dict[str, str]:
        """
        Execute task through workflow of agents.
        
        Args:
            task: Initial task
            workflow: List of agent names in execution order
            
        Returns:
            Dict of agent responses
        """
        results = {}
        current_task = task
        
        for agent_name in workflow:
            if agent_name not in self.agents:
                self.log_error(f"Agent not found: {agent_name}")
                continue
            
            agent = self.agents[agent_name]
            result = agent.run(current_task)
            results[agent_name] = result
            
            # Use result as input for next agent
            current_task = f"Previous result from {agent_name}:\n{result}\n\nOriginal task: {task}"
        
        return results


# Example usage
def example_simple_agent():
    """Example: Simple agent"""
    agent = SimpleAgent(
        name="Analyst",
        system_prompt="You are a data analyst. Analyze data and provide insights."
    )
    
    result = agent.run(
        "Analyze this sales data: Q1: 100k, Q2: 150k, Q3: 140k, Q4: 180k"
    )
    print(f"Agent response:\n{result}")


def example_multi_agent():
    """Example: Multi-agent system"""
    # Create agents
    researcher = SimpleAgent(
        name="Researcher",
        system_prompt="You gather and summarize information."
    )
    
    analyst = SimpleAgent(
        name="Analyst",
        system_prompt="You analyze data and draw conclusions."
    )
    
    writer = SimpleAgent(
        name="Writer",
        system_prompt="You write clear, professional reports."
    )
    
    # Create system
    system = MultiAgentSystem()
    system.add_agent(researcher)
    system.add_agent(analyst)
    system.add_agent(writer)
    
    # Execute workflow
    results = system.execute_workflow(
        task="Create a report about AI adoption in Norway",
        workflow=["Researcher", "Analyst", "Writer"]
    )
    
    print("\nMulti-agent workflow results:")
    for agent_name, result in results.items():
        print(f"\n{agent_name}:\n{result[:200]}...")


if __name__ == "__main__":
    print("=== Simple Agent ===")
    example_simple_agent()
    
    print("\n=== Multi-Agent System ===")
    example_multi_agent()
