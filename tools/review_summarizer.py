from src.custom_llm import CustomLLMModel

class ReviewSummarizer():

    def __init__(self):
        self.SUMMARY_PROMPT = """
            Summarize the following text in a way that preserves all key facts, critical information, and 
            important context.Do not oversimplify or omit essential details.
            Retain names, numbers, dates, and causal relationships.
            Rephrase concisely but accurately reflect the original meaning.
            The output should be shorter than the original, yet complete in substance.
            Text to summarize:"""

        self.chat = CustomLLMModel().getchatinstance()

    def getSummaries(self, reviews) -> list:
        summarized_reviews = [(self.chat.invoke(input=f"{self.SUMMARY_PROMPT} {review}")).content for review in reviews]
        return summarized_reviews

