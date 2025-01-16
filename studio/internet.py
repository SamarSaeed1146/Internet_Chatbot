from langchain_google_genai import ChatGoogleGenerativeAI

llm: ChatGoogleGenerativeAI = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    max_retries=2,
    temperature=0.5,
)

from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)
tools = [tool]

from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

session_memory = {}

def internet_problem_solver(user_input: str):
    if "internet" not in user_input.lower():
        return "I can only assist with internet-related problems. Please ask about issues such as slow speeds, connection drops, or Wi-Fi setups."

    responses = []
    try:
        tool_response = tool.invoke(user_input)

        if tool_response:
            responses.append(f"{tool_response}")
    except Exception as e:
        print(f"Error invoking TavilySearchResults: {str(e)}")
        responses.append(f"Error with TavilySearchResults: {str(e)}")

    if responses:
        return "\n\n".join(responses)
    else:
        return "No meaningful response from tools or LLM."

def greeting_handler(user_input: str):
    greetings = ["hello", "hi", "hey", "greetings"]
    if any(greet in user_input.lower() for greet in greetings):
        return f"{user_input.capitalize()}! How can I assist you today? If you have any internet-related issues, feel free to ask!"
    else:
        return "Redirecting to the Internet Problem Solver for your query."

def is_exit_command(user_input: str):
    exit_commands = ["quit", "exit", "q", "bye", "goodbye", "see you"]
    return any(cmd in user_input.lower() for cmd in exit_commands)

class State(TypedDict):
    messages: Annotated[list, add_messages]
    
graph_builder = StateGraph(State)
graph_builder.add_node("Greeting", greeting_handler)
graph_builder.add_node("Internet Problem Solver", internet_problem_solver)
graph_builder.add_edge(START, "Greeting")
graph_builder.add_conditional_edges(
    "Greeting",
    {
        "Internet Problem Solver": lambda user_input: "internet" in user_input.lower(),
        END: lambda user_input: not ("internet" in user_input.lower())
    }
)
graph = graph_builder.compile(checkpointer=memory)

while True:
    try:
        user_input = input("User: ").strip()
        if is_exit_command(user_input):
            print("Goodbye! Feel free to return anytime for Internet Problem Solving query.")
            break

        session_memory['last_query'] = user_input

        answer = internet_problem_solver(user_input) if "internet" in user_input.lower() else greeting_handler(user_input)
        print(f"Assistant: {answer}")
    except Exception as e:
        print(f"Error: {e}")