Files:
- openapi-response-format-orig.py: Original code from DL course, converted from ipython code
- openapi-response-format.py: Use response-format to tell Open AI LLM to return the response in a format using Pydantic data
- outlines-samples.py: Various way to use outlines to modify response format from an open source model: HuggingFaceTB/SmolLM2-135M-Instruct
- outline-samples-orig.py: original code exported from iPython.
- hotdog-test.py: Use a custom model to response to image

Reading notes:
- This is OPEN AI's beta API release to support Pydantic structured output
  * Look for "response_format=Mention", it tells LLM to output using Mention structure.
- not clear if LangChain support it or not.

Extra installation:

- Install all the dependency for "sentencepiece" required by the transformer packages
  $python -m pip install "transformers[sentencepiece]"

 - Dataset: Hugging Face data set
 # inside your activated .venv
python -m pip install datasets

