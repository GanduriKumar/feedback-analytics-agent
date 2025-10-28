from src.utilities.custom_llm import CustomLLMModel
import pandas as pd
import os, dotenv, csv
from pydantic import BaseModel


class Sentiment(BaseModel):
    sentiment: str


class Summary(BaseModel):
    summary: str


class TestCUJ(BaseModel):
    test_user_journey: str


class UserReview(BaseModel):
    user_review: str


class Category(BaseModel):
    category: str


class Output(BaseModel):
    sentiment: Sentiment
    category: Category
    user_review: UserReview
    summary: Summary
    test_user_journey: TestCUJ


class ReviewClassifier:
    """

    """

    def __init__(self):
        dotenv.load_dotenv()
        self.model = CustomLLMModel()
        # self.client = self.model.getclientinterface()
        self.MODEL = os.getenv('INFERENCE_MODEL')
        self.classification_labels = ["Audio","Watch","Bluetooth", "Wi-Fi", "CarKit", "Other"]
        self.classification_guidelines = (
                f"Categorize the review into exactly one of labels in {self.classification_labels} "
                f"** Strictly ensuring ** : "
                f"1. No new lines or extra white spaces."
                f"2. No additional words, explanations, or qualifiers."
                f"3. When no relevant mapping to the provided labels is detected, use Other"
            )

        self.classification_task = (
                f"You are an expert in choosing the most appropriate label for classifying a given text "
                f"and do not deviate from the guidelines "
            )
        self.testCUJ_task = (
                f"You are a senior, experienced software quality analyst and tester specializing "
                f"in mobile phones and accessories. Generate clear and concise instructions to create a "
                f"test user journey using language and descriptions that a tester can easily understand and follow"
                f"that addresses the key issue described in the review"
            )
        self.summarization_task = (
            f"You are a world-class expert in text summarization, with a keen ability to distill "
            f"complex information into its most essential elements. Your task is to analyze the "
            f"given review and create a summary that captures its core message and most significant "
            f"points in exactly two concise, meaningful, and well-crafted sentences. "
            f"Ensure that your summary is both comprehensive and succinct, leaving no crucial information "
            f"out while avoiding unnecessary details."
            )

    def classifyReviews(self, sentiment: str):
        """
        """
        try:
            df = pd.read_json(f"./reddit_{sentiment}_reviews.json")
            df = df.astype(str)
            comment_list = [df['user_review'][record] for record in range(0, df['user_review'].size)]
            print(f"Classification of {sentiment} reviews in progress")
            comment_classification = self.classifyAndUpdate(
                comment_list=comment_list,
                sentiment=sentiment
            )

            print(f"Classification of {sentiment} reviews complete ")
            if comment_classification:
                self.saveToFile(
                    sentiment=sentiment,
                    comment_classification=comment_classification
                )
            else:
                print(" No reviews found!")

        except Exception as e:
            print(f"Error fetching reviews: {e}")

    def classifyAndUpdate(self, comment_list: list, sentiment: str):
        """
        """
        classifications = []
        # print("Entering classification")
        for comment in comment_list:
            client = self.model.getclientinterface()
            # print("Classification started")
            classifier = client.generate(
                model=self.MODEL,
                prompt=(f"Perform the task in {self.classification_task} on {comment} adhering "
                        f"to guidelines in {self.classification_guidelines}")
            )
            sentiment = sentiment
            summarizer = client.generate(
                model=self.MODEL,
                prompt=f"perform the task in {self.summarization_task} in {comment}"
            )
            testCUJ = client.generate(
                model=self.MODEL,
                prompt=f"perform the task in {self.testCUJ_task} in {comment}"
            )
            # print("Classification done")
            classifications.append({
                "sentiment": sentiment,
                "categories": [classifier.response],
                "user_review": comment,
                "summary": [summarizer.response],
                "test_user_journey": [testCUJ.response]
            })
            # print("Updates done")
        return classifications

    def saveToFile(self, sentiment: str, comment_classification: list):
        """
        """
        print(comment_classification)
        df = pd.DataFrame(comment_classification)
        json_file_name = f"reddit_{sentiment}_review_classification.json"
        df.to_json(json_file_name,indent=4,orient="records")

        csv_file_name = f"reddit_{sentiment}_review_classification.csv"
        df.to_csv(csv_file_name, index=False, quoting=csv.QUOTE_ALL, quotechar='"')

    def classifyReview(self, comment: str, sentiment: str):
        """
        """
        classification = {}
        # print("Entering classification")
        # for comment in comment_list:
        model = CustomLLMModel()
        client = model.getclientinterface()
        # print("Classification started")
        # classifier = client.generate(
        #     model=self.MODEL,
        #     prompt=(f"Perform the task in {self.classification_task} on {comment} adhering "
        #             f"to guidelines in {self.classification_guidelines}"),
        # )
        sentiment = sentiment
        # print("Classification done")
        summarizer = client.generate(
            model=self.MODEL,
            prompt=f"perform the task in {self.summarization_task} in {comment}"
        )
        # testCUJ = client.generate(
        #     model=self.MODEL,
        #     prompt=f"perform the task in {self.testCUJ_task} in {comment}"
        # )
        classification = {
            "sentiment": sentiment,
            # "categories": classifier.response,
            "user_review": comment,
            "summary": summarizer.response,
            # "test_user_journey": testCUJ.response
        }
        print(f"Updates done for:\n {summarizer.response}")
        return classification

