import pandas as pd
import dotenv, csv
from src.custom_llm import CustomLLMModel



class GenerateSearchQueries:

    def __init__(self):
        dotenv.load_dotenv()
        self.model = CustomLLMModel()
        self.client = self.model.getclientinterface()
        self.keyphrases = (
                f"Pixel vs iPhone comparison"
                f"Challenges switching from iPhone to Pixel"
                f"Pixel ecosystem advantages over iPhone ecosystem"
                f"Google ecosystem vs Apple ecosystem"
                f"Pixel features compared to iPhone features"
                f"Pixel ecosystem vs iPhone ecosystem user reviews"
               )

    def generateQueries(self):
        df = pd.read_csv("inputfile.csv", header=0, delimiter=",")
        query = self.client.generate(
            model=self.model.MODEL_NAME,
            prompt = (
                f"You are an expert in crafting targeted Reddit search queries to extract comprehensive user feedback"
                f"Generate a complete list of natural-language search phrases that combine these elements from our "
                f"dataset:"
                f"1. **Device Combinations** (from {df['1P Devices  Combination']}, REQUIRED)"
                f"2. **Core User Pain Points** (from {df['Critical User Journey Problem area']}, REQUIRED)"
                f"3. **Hardware/Software Components** (from {df['Component']}, REQUIRED)"
                f"4. **Functionality** (from{df['Functionality']}, REQUIRED)"
                f"5. **Comparison with similar competition products (from Apple), REQUIRED"

                f"**Requirements:**"
                f"- Mandatory inclusion of Critical User Journey Problem area"
                f"- Exclude any entry with missing values (NaNs) in ANY field"
                f"- Create 20-30 word natural phrases, not keyword lists"
                f"- Blend elements contextually (e.g., '[Device] [Problem] during [Feature] usage due to [Component]')"
                f"- Output as clean text with one query per line without numbering"
                f"- No markdown, numbering, or special characters"
                f"- Include relevant Apple product(s) to extract comparitive feedback"

                f"Example structures using key phrases in {self.keyphrases}:"
                f"1. Call conferencing issues in Pixel ecosystem vs Apple ecosystem for [Device Combination]. "
                f"2. [Problem] caused when using Pixel products Vs Apple products for [Device Combination]"
        )
    )
        search_queries = query.response.splitlines()
        queries = []
        for search_query in search_queries:
            queries.append({
                "queries": search_query
            })
            print(search_query)

        df = pd.DataFrame(queries)
        df = df.astype(str)
        df.to_json("./search_queries.json", index=False)
        df.to_csv("./search_queries.csv", index=False,quoting=csv.QUOTE_ALL)
