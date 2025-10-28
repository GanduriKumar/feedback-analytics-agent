from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

def getSummaries(reviews: list) -> list:
    """
        Summarize the given list of reviews
        Args:
            reviews: List of reviews
        Return:
            list: List of summarized reviews
    """
    return [getSummary(review) for review in reviews]
    # for review in reviews:
    #     summarizer = LsaSummarizer()
    #     parser = PlaintextParser.from_string(review, Tokenizer("english"))
    #     # summary_first_pass = summarizer(parser.document, 1)
    #     # parser = PlaintextParser.from_string(summary_first_pass, Tokenizer("english"))# 1 sentences
    #     summary = summarizer(parser.document, 1)
    #     summaries.append(summary)
    # return summaries

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
