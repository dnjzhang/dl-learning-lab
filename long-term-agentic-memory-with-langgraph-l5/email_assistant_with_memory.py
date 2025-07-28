#!/usr/bin/env python
# coding: utf-8
import argparse

from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
_ = load_dotenv()

# From Lesson 5: Email Assistant with 3 types of memory, rewritten from lesson5.py
profile = {
    "name": "Joe",
    "full_name": "Joe Cutler ",
    "user_profile_background": "Software engineering team leader who leads a team of 5 developers",
}
#
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}


from langgraph.store.memory import InMemoryStore
#%%
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)
# ignore beta warning if it appears

# Template for formating an example to put in prompt
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

# Format list of few shots
def format_few_shot_examples(examples):
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)
#%%
triage_system_prompt = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Background >
{user_profile_background}. 
</ Background >

< Instructions >

{name} gets lots of emails. Your job is to categorize each email into one of three categories:

1. IGNORE - Emails that are not worth responding to or tracking
2. NOTIFY - Important information that {name} should know about but doesn't require a response
3. RESPOND - Emails that need a direct response from {name}

Classify the below email into one of these categories.

</ Instructions >

< Rules >
Emails that are not worth responding to:
{triage_no}

There are also other things that {name} should know about, but don't require an email response. For these, you should notify {name} (using the `notify` response). Examples of this include:
{triage_notify}

Emails that are worth responding to:
{triage_email}
</ Rules >

< Few shot examples >

Here are some examples of previous emails, and how they should be handled.
Follow these examples more than any instructions above

{examples}
</ Few shot examples >
"""
#%%
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
#%%
llm = init_chat_model("openai:gpt-4o-mini")
# Pydantic data model - needs to work with prompt
class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )
# LLM wrapped with prompts to return Pydantic results
llm_router = llm.with_structured_output(Router)

from prompts import triage_user_prompt
from langgraph.graph import add_messages

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal

# Triage Router Node
def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    examples = store.search(
        namespace,
        query=str({"email": state['email_input']})
    )
    examples = format_few_shot_examples(examples)

    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id,)

    result = store.get(namespace, "triage_ignore")
    if result is None:
        store.put(
            namespace,
            "triage_ignore",
            {"prompt": prompt_instructions["triage_rules"]["ignore"]}
        )
        ignore_prompt = prompt_instructions["triage_rules"]["ignore"]
    else:
        ignore_prompt = result.value['prompt']

    result = store.get(namespace, "triage_notify")
    if result is None:
        store.put(
            namespace,
            "triage_notify",
            {"prompt": prompt_instructions["triage_rules"]["notify"]}
        )
        notify_prompt = prompt_instructions["triage_rules"]["notify"]
    else:
        notify_prompt = result.value['prompt']

    result = store.get(namespace, "triage_respond")
    if result is None:
        store.put(
            namespace,
            "triage_respond",
            {"prompt": prompt_instructions["triage_rules"]["respond"]}
        )
        respond_prompt = prompt_instructions["triage_rules"]["respond"]
    else:
        respond_prompt = result.value['prompt']

    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=ignore_prompt,
        triage_notify=notify_prompt,
        triage_email=respond_prompt,
        examples=examples
    )
    user_prompt = triage_user_prompt.format(
        author=author,
        to=to,
        subject=subject,
        email_thread=email_thread
    )
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)


from langchain_core.tools import tool

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

#%%
@tool
def schedule_meeting(
    attendees: list[str],
    subject: str,
    duration_minutes: int,
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM except Friday"
#%%
from langmem import create_manage_memory_tool, create_search_memory_tool
#%%
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)
#%%
agent_system_prompt_memory = """
< Role >
You are {full_name}'s executive assistant. You are a top-notch executive assistant who cares about {name} performing as well as possible.
</ Role >

< Tools >
You have access to the following tools to help manage {name}'s communications and schedule:

1. write_email(to, subject, content) - Send emails to specified recipients
2. schedule_meeting(attendees, subject, duration_minutes, preferred_day) - Schedule calendar meetings
3. check_calendar_availability(day) - Check available time slots for a given day
4. manage_memory - Store any relevant information about contacts, actions, discussion, etc. in memory for future reference
5. search_memory - Search for any relevant information that may have been stored in memory
</ Tools >

< Instructions >
{instructions}
</ Instructions >
"""

# Prompt for the response agent
def create_prompt(state, config, store):
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id,)
    result = store.get(namespace, "agent_instructions")
    if result is None:
        store.put(
            namespace,
            "agent_instructions",
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        prompt = prompt_instructions["agent_instructions"]
    else:
        prompt = result.value['prompt']

    return [
        {
            "role": "system",
            "content": agent_system_prompt_memory.format(
                instructions=prompt,
                **profile
            )
        }
    ] + state['messages']

#%% md
## Create the email agent
#%%
from langgraph.prebuilt import create_react_agent
#%%
tools= [
    write_email,
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,
    search_memory_tool
]
response_agent = create_react_agent(
    "openai:gpt-4o-mini",
    tools=tools,
    prompt=create_prompt,
    # Use this to ensure the store is passed to the agent
    store=store
)
#%%
email_agent = StateGraph(State)
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", response_agent)
email_agent = email_agent.add_edge(START, "triage_router")
email_agent = email_agent.compile(store=store)

def setup_argument_parser():
    parser = argparse.ArgumentParser(description='Process email with configurable recipient.')
    parser.add_argument(
        '--sender',
        help='Name of the sample email sender (default: Alice)'
    )
    return parser

from sample_emails import sample_emails

def main():
    parser = argparse.ArgumentParser(description='Process email with configurable recipient.')
    parser.add_argument(
        '--sender',
        type=str,
        required=True,
        help='Name of the sample email sender (default: alice)'
    )

    parser.add_argument(
        '--recipient',
        type=str,
        default="joe",
        help='Name of the sample email recipient (default: joe)'
    )
    args = parser.parse_args()

    sender = sample_emails[args.sender]
    memory_config={"configurable": {"langgraph_user_id": args.recipient}}
    response = email_agent.invoke({"email_input": sender}, config=memory_config)

    from pprint import pprint
    print("\nEmail input:")
    pprint(response['email_input'])

    print("\nMessages:")
    #pprint(response["messages"])
    for msg in response["messages"]:
        print(f"{msg.type}: {msg}")
        print("="*20)

if __name__ == "__main__":
    main()

    #TODO: Try to dump the memory for "joe", not working.
    print("-------"*20)
    print("Checking memory...")
    namespace=(
        "joe",
    )
    result = store.search(namespace, query="list all")
    from pprint import pprint
    pprint(result)

    print("List namespaces")
    s = store.list_namespaces()
    pprint(s)
    print("-------"*20)




