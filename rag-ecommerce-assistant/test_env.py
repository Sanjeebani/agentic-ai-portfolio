import os
from dotenv import load_dotenv

load_dotenv()

print("API KEY FOUND:", bool(os.getenv("OPENAI_API_KEY")))