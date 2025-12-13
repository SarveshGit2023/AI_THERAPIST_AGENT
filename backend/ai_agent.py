try:
    from langchain.tools import tool
except Exception:
    try:
        from langchain.agents import tool
    except Exception:
        def tool(func=None, **kwargs):
            if func is None:
                def decorator(f):
                    return f
                return decorator
            return func
from .tools import query_medgemma, call_emergency

@tool
def ask_mental_health_specialist(query: str) -> str:
    """
    Generate a therapeutic response using the MedGemma model.
    Use this for all general user queries, mental health questions, emotional concerns,
    or to offer empathetic, evidence-based guidance in a conversational tone.
    """
    return query_medgemma(query)


@tool
def emergency_call_tool() -> None:
    """
    Place an emergency call to the safety helpline's phone number via Twilio.
    Use this only if the user expresses suicidal ideation, intent to self-harm,
    or describes a mental health emergency requiring immediate help.
    """
    call_emergency()


@tool
def locate_therapist_tool(location: str) -> str:
    """
    Finds and returns a list of licensed therapists near the specified location using OpenStreetMap.

    Args:
        location (str): The name of the city or area in which the user is seeking therapy support.

    Returns:
        str: A newline-separated string containing therapist names, addresses, and contact info.
    """
    import requests

    try:
        # Search for therapists/mental health facilities near the location
        queries = [
            f"therapist {location}",
            f"psychologist {location}",
            f"mental health clinic {location}",
            f"counseling center {location}"
        ]

        all_results = []
        seen_names = set()

        for query in queries:
            params = {
                'q': query,
                'format': 'json',
                'limit': 3,
                'addressdetails': 1,
                'extratags': 1
            }

            response = requests.get('https://nominatim.openstreetmap.org/search',
                                  params=params,
                                  headers={'User-Agent': 'SafeSpace-AI-Therapist/1.0'})

            if response.status_code == 200:
                results = response.json()
                for result in results:
                    name = result.get('display_name', '').split(',')[0]
                    if name and name not in seen_names and len(name) > 3:  # Filter out very short names
                        seen_names.add(name)

                        # Extract structured address information
                        address_details = result.get('address', {})
                        house_number = address_details.get('house_number', '')
                        road = address_details.get('road', '')
                        city = address_details.get('city', address_details.get('town', address_details.get('village', '')))
                        state = address_details.get('state', '')
                        postcode = address_details.get('postcode', '')

                        # Build a clean address
                        address_parts = []
                        if house_number and road:
                            address_parts.append(f"{house_number} {road}")
                        elif road:
                            address_parts.append(road)

                        if city:
                            address_parts.append(city)
                        if state:
                            address_parts.append(state)
                        if postcode:
                            address_parts.append(postcode)

                        full_address = ", ".join(address_parts) if address_parts else result.get('display_name', 'Address not available')

                        # Get coordinates for mapping
                        lat = result.get('lat', '')
                        lon = result.get('lon', '')

                        # Format as a detailed entry
                        entry = f"**{name}**\n"
                        entry += f"📍 Address: {full_address}\n"
                        entry += f"📞 Phone: Contact facility directly or search online directories\n"
                        if lat and lon:
                            entry += f"🗺️ Coordinates: {lat}, {lon}\n"
                        entry += "---"

                        all_results.append(entry)

                        if len(all_results) >= 5:  # Limit to 5 results total
                            break

            if len(all_results) >= 5:
                break

        if all_results:
            header = f"Here are mental health professionals and clinics I found near {location}:\n\n"
            footer = "\n\n💡 **Note**: Phone numbers aren't available through public maps. I recommend:\n"
            footer += "• Calling the facility directly using the address\n"
            footer += "• Searching online directories like Psychology Today or Healthgrades\n"
            footer += "• Contacting your local mental health association\n"
            footer += "• Using Google Maps or Yelp for additional contact information"

            return header + "\n\n".join(all_results) + footer
        else:
            return f"I couldn't find specific therapist listings near {location} using OpenStreetMap. Please consider searching for local mental health directories or contacting your local health department for professional referrals."

    except Exception as e:
        return f"I'm having trouble accessing location services right now. For immediate help, please contact your local mental health crisis hotline or search for therapists in {location} through online directories."


# Step1: Create an AI Agent & Link to backend
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from .config import OPENAI_API_KEY


tools = [ask_mental_health_specialist, emergency_call_tool, locate_therapist_tool]
llm = ChatOpenAI(model="gpt-4", temperature=0.2, api_key=OPENAI_API_KEY)
graph = create_react_agent(llm, tools=tools)

SYSTEM_PROMPT = """
You are an AI engine supporting mental health conversations with warmth and vigilance.
You have access to three tools:

1. `ask_mental_health_specialist`: Use this tool to answer all emotional or psychological queries with therapeutic guidance.
2. `locate_therapist_tool`: Use this tool if the user asks about nearby therapists or if recommending local professional help would be beneficial.
3. `emergency_call_tool`: Use this immediately if the user expresses suicidal thoughts, self-harm intentions, or is in crisis.

Always take necessary action. Respond kindly, clearly, and supportively.
"""

def parse_response(stream):
    tool_called_name = "None"
    final_response = None

    for s in stream:
        # Check if a tool was called
        tool_data = s.get('tools')
        if tool_data:
            tool_messages = tool_data.get('messages')
            if tool_messages and isinstance(tool_messages, list):
                for msg in tool_messages:
                    tool_called_name = getattr(msg, 'name', 'None')

        # Check if agent returned a message
        agent_data = s.get('agent')
        if agent_data:
            messages = agent_data.get('messages')
            if messages and isinstance(messages, list):
                for msg in messages:
                    if msg.content:
                        final_response = msg.content

    return tool_called_name, final_response


"""if __name__ == "__main__":
    while True:
        user_input = input("User: ")
        print(f"Received user input: {user_input[:200]}...")
        inputs = {"messages": [("system", SYSTEM_PROMPT), ("user", user_input)]}
        stream = graph.stream(inputs, stream_mode="updates")
        tool_called_name, final_response = parse_response(stream)
        print("TOOL CALLED: ", tool_called_name)
        print("ANSWER: ", final_response)"""
        