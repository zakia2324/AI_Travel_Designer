import os
from dotenv import load_dotenv
from typing import cast
import chainlit as cl
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, handoff
from agents.run import RunConfig, RunContextWrapper

# Load the environment variables from the .env file
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

# Check if the API key is present; if not, raise an error
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file.")

# Define the on_handoff callback function





@cl.on_chat_start
async def start():
    external_client = AsyncOpenAI(
        api_key=os.getenv("GEMINI_API_KEY"),
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = OpenAIChatCompletionsModel(
        model="gemini-2.5-flash-lite-preview-06-17",
        openai_client=external_client
    )

    config = RunConfig(
        model=model,
        model_provider=external_client,
        tracing_disabled=True
    )


    def on_handoff(agent: Agent, ctx: RunContextWrapper[None]):
        agent_name = agent.name
        print("--------------------------------")
        print(f"Handing off to {agent_name}...")
        print("--------------------------------")
        # Send a more visible message in the chat
        cl.Message(
            content=f"🔄 **Handing off to {agent_name}...**\n\nI'm transferring your request to our {agent_name.lower()} who will be able to better assist you.",
            author="System"
        ).send()

    
    destination_agent = Agent(name="Destination Agent", instructions="You are a destination agent,you find nice places for the visitors.", model=model,tools=[])
    booking_agent = Agent(name="Booking Agent", instructions="You are a booking agent,you book flights for people", model=model,tools=[])

    # Correct on_handoff function definition
    

    agent = Agent(
        name="Triage Agent",
        instructions="You are a triage agent",
        model=model,
        handoffs=[
            handoff(destination_agent, on_handoff=lambda ctx: on_handoff(destination_agent, ctx)),
            handoff(booking_agent, on_handoff=lambda ctx: on_handoff(booking_agent, ctx))
        ]
    )

    # Set session variables
    cl.user_session.set("agent", agent)
    cl.user_session.set("config", config)
    cl.user_session.set("destination_agent", destination_agent)
    cl.user_session.set("booking_agent", booking_agent)
    cl.user_session.set("chat_history", [])

    await cl.Message(content="Hello welcome to AI Designer Travel Agent,How may i help you ?").send()


@cl.on_message
async def main(message: cl.Message):
    """Process incoming messages and generate responses."""

    
    # Send a thinking message
    msg = cl.Message(content="Thinking...")
    await msg.send()

    agent: Agent = cast(Agent, cl.user_session.get("agent"))
    config: RunConfig = cast(RunConfig, cl.user_session.get("config"))

    # Retrieve the chat history from the session.
    history = cl.user_session.get("chat_history") or []

    # Append the user's message to the history.
    history.append({"role": "user", "content": message.content})

    # Check if the message is about flight booking and trigger handoff
    if any(word in message.content.lower() for word in ["flight", "book flight", "flight booking", "plane ticket", "air ticket"]):
        booking_agent = cl.user_session.get("booking_agent")
        if booking_agent:
            def on_handoff(agent, ctx):
                agent_name = agent.name
                print("--------------------------------")
                print(f"Handing off to {agent_name}...")
                print("--------------------------------")
                cl.Message(
                    content=f"🔄 **Handing off to {agent_name}...**\n\nI'm transferring your request to our {agent_name.lower()} who will be able to better assist you.",
                    author="System"
                ).send()
            on_handoff(booking_agent, None)
            cl.user_session.set("agent", booking_agent)
            msg.content = "You are now being transferred to the Booking Agent. Please continue with your flight booking request."
            await msg.update()
            return

    # Check if the message is about destination suggestion and trigger handoff
    if any(word in message.content.lower() for word in ["destination", "suggest destination", "where to go", "holiday place", "travel place", "visit", "tourist spot", "recommend place"]):
        destination_agent = cl.user_session.get("destination_agent")
        if destination_agent:
            def on_handoff(agent, ctx):
                agent_name = agent.name
                print("--------------------------------")
                print(f"Handing off to {agent_name}...")
                print("--------------------------------")
                cl.Message(
                    content=f"🔄 **Handing off to {agent_name}...**\n\nI'm transferring your request to our {agent_name.lower()} who will be able to better assist you.",
                    author="System"
                ).send()
            on_handoff(destination_agent, None)
            cl.user_session.set("agent", destination_agent)
            msg.content = "You are now being transferred to the Destination Agent. Please continue with your destination request."
            await msg.update()
            return

    try:
        result = Runner.run_sync(agent, history, run_config=config)

        response_content = result.final_output

        # Update the thinking message with the actual response
        msg.content = response_content
        await msg.update()

        # IMPORTANT FIX HERE: use "developer" instead of "assistant"
        history.append({"role": "developer", "content": response_content})

        # Update session history
        cl.user_session.set("chat_history", history)
        print(f"History: {history}")

    except Exception as e:
        msg.content = f"Error: {str(e)}"
        await msg.update()
        print(f"Error: {str(e)}")

#         def main():
#                       print("Hello from career-mentore!")


# if __name__ == "__main__":
#     main()