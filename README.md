# Gmail & Calendar AI Assistant

A conversational AI assistant that manages your Gmail and Google Calendar through natural language. Built with LangChain, LangGraph, and Ollama.

### Gmail Agent
- **Draft emails** - Compose email drafts with natural language instructions
- **Search emails** - Find emails using Gmail's search syntax
- **Organize emails** - Automatically categorize and label emails
- **Read email content** - Retrieve and analyze email details

### Calendar Agent
- **Create events** - Schedule meetings and appointments
- **Search events** - Find events by text or time range
- **Update events** - Modify existing calendar entries
- **Delete events** - Remove unwanted events
- **Check availability** - See your free/busy times

### Orchestrator
- **Natural conversation** - Chat about any topic
- **Smart delegation** - Automatically routes Gmail and Calendar tasks to the right agent
- **Context awareness** - Maintains conversation history
- **General knowledge** - Answers questions beyond Gmail and Calendar

## How It Works

The system uses a multi-agent architecture:
1. The **Orchestrator** handles general conversation and identifies when to delegate tasks
2. Specialized agents process Gmail or Calendar requests using Google APIs
3. Results are presented back in a conversational format

Built with:
- **LangChain** for agent framework
- **LangGraph** for stateful orchestration
- **Ollama** for local LLM inference (using qwen3:8b model)
- **Google APIs** for Gmail and Calendar integration