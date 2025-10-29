from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def getSummaries(reviews: list) -> list:
    """
    Summarizes a list of user reviews.

    Args:
        reviews (list): A list of reviews to be summarized.

    Returns:
        list: A list containing the summarized reviews.
    """
    return [getSummary(review) for review in reviews]
 
def getSummary(review: str) -> str:
    """
        Summarize a given review
        Args:
            review: End user review
        Return:
            str: summarized review
    """
    summarizer = LsaSummarizer()
    parser = PlaintextParser.from_string(review, Tokenizer("english"))
    summary = summarizer(parser.document, 1)
    summary_text = ' '.join([str(sentence) for sentence in summary])
    return summary_text
