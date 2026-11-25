import os
import warnings
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import PydanticOutputParser

from tools import (
    get_air_quality,
    get_weather_and_wind,
    calculate_cigarettes,
    air_diary,
)

warnings.filterwarnings("ignore")

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("Missing GEMINI_API_KEY in .env file.")

if not os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=api_key,
    disable_streaming=True,
    max_retries=5,
)

TOOLS = [
    get_air_quality,
    get_weather_and_wind,
    calculate_cigarettes,
    air_diary,
]

SYSTEM_PROMPT = """You are TazaAir, a smart Air Quality and Smog Shield assistant for cities and mountain spots in Kazakhstan (Almaty, Astana, Karaganda, Ust-Kamenogorsk, Medeu, Shymbulak, Kok-Zhailau, Big Almaty Lake, Burabay) and worldwide.

Strict rules:
1. NEVER guess numbers. Always call your tools to get real air quality, weather/wind, and cigarette calculations.
2. When asked about air quality or going outside, call `get_air_quality` (and `get_weather_and_wind` when helpful) and `calculate_cigarettes`.
3. Call `air_diary` ONLY when the user asks to save a note or read saved history.
4. Reply strictly in the language of the user's latest message (Russian or English).
5. Format cleanly for a plain terminal console: use simple '-' bullet points and do NOT use markdown asterisks (** or *)."""

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


def extract_text(raw_output) -> str:
    if isinstance(raw_output, list):
        raw_output = "\n".join(
            item["text"] if isinstance(item, dict) and "text" in item else str(item)
            for item in raw_output
        )
    return str(raw_output).replace("**", "").strip()


def normalize_output(result: dict) -> dict:
    result["output"] = extract_text(result.get("output", ""))
    return result


agent = create_tool_calling_agent(llm=llm, tools=TOOLS, prompt=agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=TOOLS,
    return_intermediate_steps=True,
    handle_parsing_errors=True,
)

clean_agent_chain = agent_executor | RunnableLambda(normalize_output)

chat_store = {}


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    if session_id not in chat_store:
        chat_store[session_id] = InMemoryChatMessageHistory()
    return chat_store[session_id]


agent_with_memory = RunnableWithMessageHistory(
    clean_agent_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="output",
)


class AirSafetyCard(BaseModel):
    city: str = Field(description="City or location name")
    smog_status: str = Field(description="Short smog status: Clean, Moderate, or Heavy Smog")
    cigarettes_per_day: float = Field(description="Equivalent cigarettes inhaled per 24 hours")
    window_advice: str = Field(description="Short advice on opening or closing windows")
    mask_needed: bool = Field(description="True if an FFP2/N95 mask is recommended outside")


parser = PydanticOutputParser(pydantic_object=AirSafetyCard)

report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Extract a structured AirSafetyCard from the assistant's latest response.\n{format_instructions}",
        ),
        ("human", "{response_text}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

report_chain = report_prompt | llm | parser


def ask_agent(user_message: str, session_id: str = "default") -> dict:
    result = agent_with_memory.invoke(
        {"input": user_message},
        config={"configurable": {"session_id": session_id}},
    )
    used_tools = [step[0].tool for step in result.get("intermediate_steps", [])]
    return {"answer": result["output"], "tools_used": used_tools}


def build_safety_card(response_text: str) -> AirSafetyCard:
    return report_chain.invoke({"response_text": response_text})
