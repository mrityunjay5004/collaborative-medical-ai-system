from .agent_base import AgentBase

class ValidatorAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="ValidatorAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, topic, article):
        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant that validates research articles for accuracy and academic quality."
            },
            {
                "role": "user",
                "content": (
                    f"Topic: {topic}\n\n"
                    f"Article:\n{article}\n\n"
                    "Validate the article and rate it from 1 to 5:"
                )
            }
        ]

        return self.call_groq(
            messages=messages,
            temperature=0.3,
            max_tokens=600,
        )
