import json

class LoadProperties:
    def __init__(self, file_path='/Users/jzhang/etc/openai-config.json'):
        self.file_path = file_path
        self.properties = self.load_properties()

    def load_properties(self):
        """
        Load the properties from the JSON file.
        """
        try:
            with open(self.file_path, 'r') as json_file:
                return json.load(json_file)
        except FileNotFoundError:
            print(f"Error: {self.file_path} file not found.")
            return {}
        except json.JSONDecodeError:
            print(f"Error: Could not parse the JSON file {self.file_path}.")
            return {}

    def getApiKey(self):
        """
        Retrieve the OpenAI API key from the loaded properties.
        """
        return self.properties.get("openai-api-key", None)

    def getTavilyApiKey(self):
        return self.properties.get("tavily-api-key", None)

# Usage:
# properties = LoadProperties()
# openai_api_key = properties.getApiKey()
