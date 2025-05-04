# Add your utilities or helper functions to this file.

def print_mention(processed_mention, mention):
    # Check if we need to respond
    if processed_mention.needs_response:
        # We need to respond
        print(f"Responding to {processed_mention.sentiment} {processed_mention.product} feedback")
        print(f"  User: {mention}")
        print(f"  Response: {processed_mention.response}")
    else:
        print(f"Not responding to {processed_mention.sentiment} {processed_mention.product} post")
        print(f"  User: {mention}")

    if processed_mention.support_ticket_description:
        print(f"  Adding support ticket: {processed_mention.support_ticket_description}")
 