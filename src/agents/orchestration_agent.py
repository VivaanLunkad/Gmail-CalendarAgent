from typing import TypedDict, Annotated, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import add_messages


class OrchestrationState(TypedDict):
    """State for the orchestration workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    current_task: Optional[str]
    delegated_to: Optional[str]
    task_complete: bool


class ConversationalOrchestrator:
    """Main orchestrator for managing conversations and delegating to specialized agents"""

    def __init__(self, model_name: str = "qwen3:8b", temperature: float = 0.7):
        self.model = ChatOllama(model=model_name, temperature=temperature)
        self.sub_agents = {}
        self.memory = MemorySaver()
        self.graph = None

    def add_sub_agent(self, name: str, agent: Any, triggers: List[str]):
        """Add a subagent with trigger phrases"""
        self.sub_agents[name] = {
            "agent": agent,
            "triggers": triggers
        }

    def _get_system_prompt(self) -> str:
        """Generate system prompt based on available agents"""
        agent_descriptions = []
        for name, config in self.sub_agents.items():
            triggers = ", ".join(f"'{t}'" for t in config["triggers"])
            agent_descriptions.append(f"- {name}: handles tasks containing {triggers}")

        agents_list = "\n".join(agent_descriptions) if agent_descriptions else "No specialized agents available"

        return f"""You are a helpful conversational assistant that can chat about any topic and delegate specialized tasks to sub-agents when needed.

        Your capabilities:
        1. Have natural conversations about any topic
        2. Answer questions using your knowledge
        3. Delegate to specialized agents when users ask for specific tasks

        Available specialized agents:
        {agents_list}

        IMPORTANT: Only delegate to a specialized agent when the user explicitly asks for that specific task. 
        For general questions or conversations, respond directly.

        When you need to delegate:
        - Set the task description clearly
        - Indicate which agent should handle it
        - Let the user know you're delegating

        Otherwise, just have a friendly conversation!"""

    def _should_delegate(self, user_message: str) -> Optional[str]:
        """Check if a message should be delegated to a subagent"""
        message_lower = user_message.lower()

        for agent_name, config in self.sub_agents.items():
            for trigger in config["triggers"]:
                if trigger.lower() in message_lower:
                    return agent_name

        return None

    def orchestrator_node(self, state: OrchestrationState) -> Dict:
        """Main orchestration logic"""
        messages = state["messages"]
        last_message = messages[-1]

        # Add a system prompt if it's the first message
        system_message = [msg for msg in messages if isinstance(msg, SystemMessage)]
        if not system_message:
            messages = [SystemMessage(content=self._get_system_prompt())] + messages

        # Check if it should delegate
        if isinstance(last_message, HumanMessage):
            delegate_to = self._should_delegate(last_message.content)

            if delegate_to:
                # Create a delegation message
                agent_name = delegate_to.title()
                response_content = f"I'll help you with that {agent_name} task. Let me process your request...\n\nDelegating to {agent_name} agent..."

                return {
                    "messages": [AIMessage(content=response_content)],
                    "current_task": last_message.content,
                    "delegated_to": delegate_to,
                    "task_complete": False
                }

        # Otherwise, respond conversationally
        response = self.model.invoke(messages)

        return {
            "messages": [response],
            "current_task": None,
            "delegated_to": None,
            "task_complete": True
        }

    def create_delegation_node(self, agent_name: str):
        """Create a node for delegating to a subagent"""

        def delegation_node(state: OrchestrationState) -> Dict:
            agent = self.sub_agents[agent_name]["agent"]
            task = state.get("current_task", "")

            try:
                # Process the task with the subagent
                result = agent.process_request(task)

                response_content = f"{agent_name.title()} task completed:\n\n{result}"

                return {
                    "messages": [AIMessage(content=response_content)],
                    "current_task": None,
                    "delegated_to": None,
                    "task_complete": True
                }

            except Exception as e:
                error_content = f"Error processing {agent_name.title()} task: {str(e)}"
                return {
                    "messages": [AIMessage(content=error_content)],
                    "current_task": None,
                    "delegated_to": None,
                    "task_complete": True
                }

        return delegation_node

    def should_continue(self, state: OrchestrationState) -> str:
        """Determine the next step in the workflow"""
        if state.get("delegated_to"):
            return f"delegate_{state['delegated_to']}"
        return END

    def build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(OrchestrationState)

        # Add the main orchestrator node
        workflow.add_node("orchestrator", self.orchestrator_node)

        # Add delegation nodes for each subagent
        for agent_name in self.sub_agents:
            node_name = f"delegate_{agent_name}"
            workflow.add_node(node_name, self.create_delegation_node(agent_name))
            workflow.add_edge(node_name, END)

        # Set an entry point
        workflow.set_entry_point("orchestrator")

        # Add conditional edges
        workflow.add_conditional_edges(
            "orchestrator",
            self.should_continue,
            {
                END: END,
                **{f"delegate_{name}": f"delegate_{name}" for name in self.sub_agents}
            }
        )

        # Compile with memory
        self.graph = workflow.compile(checkpointer=self.memory)
        return self.graph

    def chat(self, message: str, thread_id: str = "default") -> str:
        """Have a conversation with the orchestrator"""
        if not self.graph:
            self.build_graph()

        initial_state = {
            "messages": [HumanMessage(content=message)],
            "current_task": None,
            "delegated_to": None,
            "task_complete": False
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(initial_state, config)

        # Return the last AI message
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        return ai_messages[-1].content if ai_messages else "No response generated"

    def start_chat(self, thread_id: str = "default"):
        """Start an interactive chat session"""
        print("🤖 Orchestrator Chat Bot Started!")
        print("=" * 50)
        print("I can help with general questions, Gmail tasks, and Calendar management.")
        print("Type 'exit', 'quit', or 'bye' to end the conversation.")
        print("=" * 50 + "\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                    print("\nBot: Goodbye! Have a great day! 👋")
                    break

                # Skip empty inputs
                if not user_input:
                    continue

                # Get response from orchestrator
                print("\nBot: ", end="", flush=True)
                response = self.chat(user_input, thread_id)

                # Print response with some formatting
                print(response)
                print()  # Add blank line for readability

            except KeyboardInterrupt:
                print("\n\nBot: Chat interrupted. Goodbye! 👋")
                break
            except Exception as e:
                print(f"\nBot: Sorry, I encountered an error: {str(e)}")
                print("Let's continue our conversation.\n")


# Factory function to create orchestrator with agents
def create_orchestrator_with_agents(gmail_agent, calendar_agent, model_name: str = "qwen3:8b") -> ConversationalOrchestrator:
    """Create an orchestrator with Gmail and Calendar capabilities"""
    orchestrator = ConversationalOrchestrator(model_name=model_name)

    # Gmail triggers - what phrases indicate Gmail tasks
    gmail_triggers = [
        "email", "gmail", "draft", "compose", "send mail",
        "search mail", "search email", "label email",
        "categorize email", "organize email", "find email"
    ]

    # Calendar triggers - what phrases indicate Calendar tasks
    calendar_triggers = [
        "calendar", "meeting", "appointment", "event", "schedule",
        "book time", "set up meeting", "create event", "add to calendar",
        "check calendar", "free time", "availability", "busy",
        "reschedule", "cancel meeting", "update event", "find meetings",
        "team standup", "recurring meeting", "all-day event"
    ]

    orchestrator.add_sub_agent("gmail", gmail_agent, gmail_triggers)
    orchestrator.add_sub_agent("calendar", calendar_agent, calendar_triggers)

    return orchestrator
