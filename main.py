from src.agents.gmail_agent import create_gmail_agent
from src.agents.calendar_agent import create_calendar_agent
from src.agents.orchestration_agent import create_orchestrator_with_agents


def run_chat_bot():
    """Run the orchestrator chatbot with Gmail and Calendar capabilities"""
    # Create agents
    gmail_agent = create_gmail_agent(model_name="qwen3:8b")
    calendar_agent = create_calendar_agent(model_name="qwen3:8b")

    # Create the orchestrator with both agents
    orchestrator = create_orchestrator_with_agents(
        gmail_agent,
        calendar_agent,
        model_name="qwen3:8b"
    )

    # Start chat
    orchestrator.start_chat()


if __name__ == "__main__":
    run_chat_bot()