def print_final_state(final_state):
    messages = final_state.get("messages", [])

    print("=" * 40)
    print("MESSAGE CONTENTS")
    print("=" * 40)
    for idx, message in enumerate(messages):
        # Determine the type of message
        message_type = type(message).__name__
        sender = "Unknown"
        if message_type == "HumanMessage":
            sender = "Human"
        elif message_type == "AIMessage":
            sender = "AI"
        elif message_type == "ToolMessage":
            sender = "Tool"

        # Extract the content
        content = getattr(message, 'content', '')
        print(f"{idx + 1}. {sender} Message:")
        print(content)
        print("-" * 40)

    print("=" * 40)
    print("TECHNICAL PARAMETERS")
    print("=" * 40)
    if messages:
        # Assuming the last message contains the most relevant technical parameters
        last_message = messages[-1]
        response_metadata = getattr(last_message, 'response_metadata', {})
        token_usage = response_metadata.get('token_usage', {})
        if token_usage:
            print("Token Usage:")
            print(f"  Input Tokens: {token_usage.get('prompt_tokens', 'N/A')}")
            print(f"  Output Tokens: {token_usage.get('completion_tokens', 'N/A')}")
            print(f"  Total Tokens: {token_usage.get('total_tokens', 'N/A')}")
            print("-" * 40)

        model_name = response_metadata.get('model_name', 'Unknown Model')
        print(f"Model Name: {model_name}")
    print("=" * 40)


def print_final_state_as_html(final_state):
    """Generate an HTML representation of the final state and print it."""
    messages = final_state.get("messages", [])

    # Start building the HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LangGraph Final State</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            .section { margin-bottom: 40px; }
            .message { border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .human { background-color: #e7f3ff; }
            .ai { background-color: #fffbe7; }
            .tool { background-color: #f3ffe7; }
            h2 { border-bottom: 2px solid #ccc; padding-bottom: 5px; }
        </style>
    </head>
    <body>
        <h1>LangGraph Final State</h1>
    """

    # Section: Message Contents
    html_content += '<div class="section"><h2>Message Contents</h2>'
    for idx, message in enumerate(messages):
        message_type = type(message).__name__
        sender = "Unknown"
        css_class = ""
        if message_type == "HumanMessage":
            sender = "Human"
            css_class = "human"
        elif message_type == "AIMessage":
            sender = "AI"
            css_class = "ai"
        elif message_type == "ToolMessage":
            sender = "Tool"
            css_class = "tool"

        content = getattr(message, 'content', '')
        html_content += f"""
        <div class="message {css_class}">
            <strong>{idx + 1}. {sender} Message:</strong>
            <p>{content or "No content available."}</p>
        </div>
        """
    html_content += "</div>"

    # Section: Technical Parameters
    html_content += '<div class="section"><h2>Technical Parameters</h2>'
    if messages:
        last_message = messages[-1]
        response_metadata = getattr(last_message, 'response_metadata', {})
        token_usage = response_metadata.get('token_usage', {})
        if token_usage:
            html_content += "<h3>Token Usage:</h3>"
            html_content += "<ul>"
            html_content += f"<li>Input Tokens: {token_usage.get('prompt_tokens', 'N/A')}</li>"
            html_content += f"<li>Output Tokens: {token_usage.get('completion_tokens', 'N/A')}</li>"
            html_content += f"<li>Total Tokens: {token_usage.get('total_tokens', 'N/A')}</li>"
            html_content += "</ul>"

        model_name = response_metadata.get('model_name', 'Unknown Model')
        html_content += f"<h3>Model Name:</h3><p>{model_name}</p>"
    html_content += "</div>"

    # End HTML
    html_content += """
    </body>
    </html>
    """

    # Print HTML content
    print(html_content)


def print_final_state_to_file_as_html(final_state, file_name="response.html"):
    """Generate an HTML representation of the final state and save it to a file."""
    messages = final_state.get("messages", [])

    # Start building the HTML content
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>LangGraph Final State</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
            .section { margin-bottom: 40px; }
            .message { border: 1px solid #ccc; padding: 10px; margin: 10px 0; border-radius: 5px; }
            .human { background-color: #e7f3ff; }
            .ai { background-color: #fffbe7; }
            .tool { background-color: #f3ffe7; }
            h2 { border-bottom: 2px solid #ccc; padding-bottom: 5px; }
        </style>
    </head>
    <body>
        <h1>LangGraph Final State</h1>
    """

    # Section: Message Contents
    html_content += '<div class="section"><h2>Message Contents</h2>'
    for idx, message in enumerate(messages):
        message_type = type(message).__name__
        sender = "Unknown"
        css_class = ""
        if message_type == "HumanMessage":
            sender = "Human"
            css_class = "human"
        elif message_type == "AIMessage":
            sender = "AI"
            css_class = "ai"
        elif message_type == "ToolMessage":
            sender = "Tool"
            css_class = "tool"

        content = getattr(message, 'content', '')
        html_content += f"""
        <div class="message {css_class}">
            <strong>{idx + 1}. {sender} Message:</strong>
            <p>{content or "No content available."}</p>
        </div>
        """
    html_content += "</div>"

    # Section: Technical Parameters
    html_content += '<div class="section"><h2>Technical Parameters</h2>'
    if messages:
        last_message = messages[-1]
        response_metadata = getattr(last_message, 'response_metadata', {})
        token_usage = response_metadata.get('token_usage', {})
        if token_usage:
            html_content += "<h3>Token Usage:</h3>"
            html_content += "<ul>"
            html_content += f"<li>Input Tokens: {token_usage.get('prompt_tokens', 'N/A')}</li>"
            html_content += f"<li>Output Tokens: {token_usage.get('completion_tokens', 'N/A')}</li>"
            html_content += f"<li>Total Tokens: {token_usage.get('total_tokens', 'N/A')}</li>"
            html_content += "</ul>"

        model_name = response_metadata.get('model_name', 'Unknown Model')
        html_content += f"<h3>Model Name:</h3><p>{model_name}</p>"
    html_content += "</div>"

    # End HTML
    html_content += """
    </body>
    </html>
    """

    # Write HTML to the output file
    with open(file_name, "w") as file:
        file.write(html_content)
    print(f"HTML document has been saved to {file_name}")






