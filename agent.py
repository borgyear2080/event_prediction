from typing import Annotated, Literal
import operator
from typing_extensions import TypedDict
from langchain.messages import AnyMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from agent_tools import AgentTools
from llm import LLMs

print("Hi")
class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int


class Agent:
    def __init__(self, model, tools, system_prompt="You are a helpful assistant."):
        self.model_with_tools = model.bind_tools(tools)
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.system_prompt = system_prompt
        self.graph = self._build_graph()

    def _llm_call(self, state: MessagesState):
        reply = self.model_with_tools.invoke(
            [
                SystemMessage(content=self.system_prompt),
                *state["messages"],
            ]
        )
        return {
            "messages": [reply],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }


    def _tool_call(self, state: MessagesState):
        tool_messages = []

        for call in state["messages"][-1].tool_calls:
            tool = self.tools_by_name[call["name"]]
            result = tool.invoke(call["args"])
            tool_messages.append(
                ToolMessage(
                    content=str(result),
                    tool_call_id=call["id"],
                )
            )

        return {"messages": tool_messages}


    def _should_continue(
        self, state: MessagesState
    ) -> Literal["tool_call", "__end__"]:
        if state["messages"][-1].tool_calls:
            return "tool_call"
        return END


    def _build_graph(self):
        builder = StateGraph(MessagesState)

        builder.add_node("llm_call", self._llm_call)
        builder.add_node("tool_call", self._tool_call)

        builder.add_edge(START, "llm_call")
        builder.add_conditional_edges("llm_call", self._should_continue)
        builder.add_edge("tool_call", "llm_call")

        return builder.compile()

    def invoke(self, messages, llm_calls=0):
        return self.graph.invoke({
            "messages": messages,
            "llm_calls": llm_calls,
        })




# Create the agent using your existing model and tools
llm1, llm2 = LLMs().get()
model = llm1  
model_2 = llm2
agent_tools = AgentTools()
tools = agent_tools.get_tools()
agent = Agent(model, tools)