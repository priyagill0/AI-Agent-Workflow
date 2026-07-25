#testing GROQ API and LANGCHAIN connection 
import os

from groq import Groq

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance Responsible AI.",
        }
    ],
    model="llama-3.3-70b-versatile",
)

print(chat_completion.choices[0].message.content)

# Now you can build chains and agents that can perform multi-step tasks. 
# This chain combines a prompt that tells the model what information to extract, 
# a parser that ensures the output follows a specific JSON format, 
# and llama-3.3-70b-versatile to do the actual text processing.

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

# Initialize Groq LLM
llm = ChatGroq(
    model_name="llama-3.3-70b-versatile",
    temperature=0.7
)

# Define the expected JSON structure
parser = JsonOutputParser(pydantic_object={
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "price": {"type": "number"},
        "features": {
            "type": "array",
            "items": {"type": "string"}
        }
    }
})

# Create a simple prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Extract product details into JSON with this structure:
        {{
            "name": "product name here",
            "price": number_here_without_currency_symbol,
            "features": ["feature1", "feature2", "feature3"]
        }}"""),
    ("user", "{input}")
])

# Create the chain that guarantees JSON output
chain = prompt | llm | parser

def parse_product(description: str) -> dict:
    result = chain.invoke({"input": description})
    print(json.dumps(result, indent=2))

        
# Example usage
description = """The Kees Van Der Westen Speedster is a high-end, single-group espresso machine known for its precision, performance, 
and industrial design. Handcrafted in the Netherlands, it features dual boilers for brewing and steaming, PID temperature control for 
consistency, and a unique pre-infusion system to enhance flavor extraction. Designed for enthusiasts and professionals, it offers 
customizable aesthetics, exceptional thermal stability, and intuitive operation via a lever system. The pricing is approximatelyt $14,499 
depending on the retailer and customization options."""

parse_product(description)

