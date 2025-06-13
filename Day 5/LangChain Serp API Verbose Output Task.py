import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, load_tools, AgentType
from langchain.callbacks.tracers import ConsoleCallbackHandler

# 1️⃣ Set your Gemini API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyCkT2xJMD4JfJOk0V5QaX3A-571ZnU7VPA"

# 🔑 Set your SerpAPI API key
# Replace "YOUR_SERPAPI_API_KEY" with your actual SerpAPI key
os.environ["SERPAPI_API_KEY"] = "2545c460f8496e0a721269ec8a107381058aa65a7c8a8ca42204a3ab29123303"


# 2️⃣ Initialize the Gemini chat model with verbose logging
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    verbose=True
)

# 3️⃣ Load built-in tools: web search & math evaluator
# The SerpAPI tool will now find the API key from the environment variable
tools = load_tools(["serpapi", "llm-math"], llm=llm)

# 4️⃣ Initialize an agent that can choose tools automatically
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

def ask_agent(query: str) -> str:
    """Runs query through the agent, printing tool usage."""
    response = agent.run(
        input=query,
        callbacks=[ConsoleCallbackHandler()]
    )
    return response

if __name__ == "__main__":
    # Example: ask a question requiring search + math
    q = "What's the population of Germany divided by 100?"
    print("=== Final Answer ===")
    print(ask_agent(q))