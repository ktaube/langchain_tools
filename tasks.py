from langchain.agents import initialize_agent
from langchain.agents import Tool
from langchain.agents import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from robocorp.tasks import task
from robocorp import vault

from RPA.Assistant.types import WindowLocation, Size
import RPA.Assistant

class ToolInputSchema(BaseModel):
    name: str = Field(description="should be a lead name")

def add_lead(params):
    """Starts the execution of the bot that fills in a new contact form in CRM with one lead company/person details."""
    print(f"Received params: {params}")
    return {"response": "Lead added successfully!"}

assistant = RPA.Assistant.Assistant()
gpt_conversation_display = []
gpt_conversation_internal = []

memory = ConversationBufferMemory(memory_key="chat_history")

addLeadTool = Tool(
    name="Add Lead Tool",
    func=add_lead,
    description="Adds a new lead to CRM, mandatory input parameters are name and email",
)

tools = [
    addLeadTool
]

secret = vault.get_secret("OpenAI")

print("OpenAI secret: ", secret)

llm = OpenAI(temperature=0, openai_api_key=secret["key"])

# replace the default prompt template by overriding the agent's llm_chain.prompt.template
# print(agent.agent.llm_chain.prompt.template)

agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=memory
)

def show_spinner():
    assistant.clear_dialog()
    assistant.add_loading_spinner(name="spinner", width=60, height=60, stroke_width=8)
    assistant.refresh_dialog()


def ask_gpt(form_data: dict):
    text = agent_chain.run(input=form_data["input"])
    gpt_conversation_display.append((form_data["input"], text))

    display_conversation()
    assistant.refresh_dialog()

def display_conversation():
    assistant.clear_dialog()
    assistant.add_heading("Conversation")

    for reply in gpt_conversation_display:
        assistant.add_text("You:", size=Size.Small)
        assistant.open_container(background_color="#C091EF", margin=2)
        assistant.add_text(reply[0])
        assistant.close_container()

        assistant.add_text("GPT:", size=Size.Small)
        assistant.open_container(background_color="#A5AACD", margin=2)
        assistant.add_text(reply[1])
        assistant.close_container()

    display_buttons()

def display_buttons():
    assistant.add_text_input("input", placeholder="Send a message", minimum_rows=3)
    assistant.add_next_ui_button("Send", ask_gpt)
    assistant.add_submit_buttons("Close", default="Close")

@task
def run_chat():

    display_conversation()

    assistant.run_dialog(
        timeout=1800, title="AI Chat", on_top=True, location=WindowLocation.Center
    )