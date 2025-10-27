from sentence_transformers import SentenceTransformer, util
import pandas as pd
import re, os, csv


#----------------reading the extracted posts into a dataframe---------------------------------------


#----------------combining the title and review text into a single text-----------------------------
# df["combined_reviews"] = df['post_title'].astype(str).str.cat(df['self_text'].astype(str), na_rep='')

class CategoryClassifier:

    def __init__(self):
        pass

    def clean_text(self, text):
        text = str(text).lower()
        text = re.sub(r'\W+', ' ', text)  # Remove punctuation
        return text

    #----------------defining themes-------------------------------------------------------------------
    def get_themes(self):
        return [
            'Call with earphones have audio issues',
            'video playing using earphones have issues',
            'conference call has issues',
            'migrating from Apple to Google has concerns',
            'Ease of using the functionality',
        ]

    #----------------creating embeddings for review and the themes-------------------------------------
    def get_sentencetransformer_model(self):
        return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    def get_theme_embeddgings(self, themes):
        return self.get_sentencetransformer_model().encode(themes)

    def encode_review(self, review):
        return self.get_sentencetransformer_model().encode(review)

    def find_similarity(self, review, theme_embedding):
        review_embedding = self.encode_review(review)
        similarities = util.pytorch_cos_sim(review_embedding, theme_embedding)
        return self.get_themes()[similarities.argmax().item()]

    #---------------find similarity and theme--------------------------------------------------------
    # Convert 'cleaned_reviews' column to a list and classify themes
    def generate_theme_mappings(self,sentiment):
        df = pd.read_csv(f"./reddit_{sentiment}_review_classification.csv")
        df['cleaned_reviews'] = df['summary'].apply(self.clean_text)
        df['category'] = [self.find_similarity(review, self.get_theme_embeddgings(self.get_themes())) for review in
                          df['cleaned_reviews']]
        df.to_csv(path_or_buf=f'./classified_{sentiment}_posts.csv', index=False, quoting=csv.QUOTE_ALL, quotechar='"')


if __name__ == "__main__":
    theme_categorizer = CategoryClassifier()
    for item in ["positive","negative","neutral"]:
        theme_categorizer.generate_theme_mappings(sentiment=item)
