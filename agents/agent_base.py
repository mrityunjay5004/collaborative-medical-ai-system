from abc import ABC, abstractmethod
from loguru import logger
import os
from dotenv import load_dotenv
from groq import Groq
import time

# Load environment variables
load_dotenv()

# Load model from env with fallback
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
FALLBACK_MODEL = "llama-3.1-8b-instant"

# Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


class AgentBase(ABC):
    def __init__(self, name, max_retries=2, verbose=True):
        self.name = name
        self.max_retries = max_retries
        self.verbose = verbose

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def call_groq(self, messages, temperature=0.7, max_tokens=1024):
        retries = 0

        while retries < self.max_retries:
            try:
                # Convert any wrong OpenAI-style content to plain text
                clean_messages = []
                for msg in messages:
                    if isinstance(msg["content"], list):
                        msg["content"] = msg["content"][0]["text"]
                    clean_messages.append(msg)

                if self.verbose:
                    logger.info(f"[{self.name}] Sending messages to Groq:")
                    for msg in clean_messages:
                        logger.debug(f"  {msg['role']}: {msg['content']}")

                try:
                    response = client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=clean_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )
                except Exception:
                    # Automatic fallback model
                    response = client.chat.completions.create(
                        model=FALLBACK_MODEL,
                        messages=clean_messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                    )

                reply = response.choices[0].message.content

                if self.verbose:
                    logger.info(f"[{self.name}] Groq response received successfully")

                return reply

            except Exception as e:
                retries += 1
                logger.error(
                    f"[{self.name}] Error during Groq call: {e}. "
                    f"Retry {retries}/{self.max_retries}"
                )
                time.sleep(1.5)

        raise Exception(
            f"[{self.name}] Failed to get response from Groq after {self.max_retries} retries."
        )
