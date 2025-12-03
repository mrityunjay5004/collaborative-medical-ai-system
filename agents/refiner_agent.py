from .agent_base import AgentBase

class RefinerAgent(AgentBase):
    def __init__(self, max_retries=2, verbose=True):
        super().__init__(name="RefinerAgent", max_retries=max_retries, verbose=verbose)

    def execute(self, draft):
        messages = [
            {
                "role": "system",
                "content": "You are an expert editor who refines and enhances research articles for clarity, coherence, and academic quality."
            },
            {
                "role": "user",
                "content": (
                    "Please refine the following research article draft:\n\n"
                    f"{draft}\n\nRefined Article:"
                )
            }
        ]

        return self.call_groq(
            messages=messages,
            temperature=0.5,
            max_tokens=2048,
        )
