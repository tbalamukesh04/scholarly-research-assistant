import json
import os
from google import genai

class TemplateGenerator:
    def __init__(self):
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        
    def generate_templates(self, n=5):
        prompt = """
        Generate 20 distinct question templates for querying academic Machine Learning papers.
        
        Rules:
            1. Use the placeholder {section} for paper sections (e.g. 'Abstract', 'Methods').
            2. Use the placeholder {topic} for technical concepts.
            3. Ensure high variance in phrasing (simple, complex, comparative).
            4. Return ONLY a JSON list of strings.
            
        Example Output:
            [
                "What does the {section} mention regarding {topic}?",
                "How does the author address {topic} in the {section}?",
                "Are there limitations mentioned in the {section} concerning {topic}?"
            ]
            """
            
        all_templates = []
        print(f"Generating templates using {n} API calls...")
        for i in range(n):
            try:
                response = self.client.models.generate_content(
                    model = "gemini-2.5-flash-lite",
                    contents = prompt, 
                    config = {"response_mime_type": "application/json"}
                )
                templates = json.loads(response.text)
                all_templates.extend(templates)
                print(f"Batch {i+1}/{n}: Got {len(templates)} templates.")
                
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                
        unique_templates = list(set(all_templates))
        
        with open("data_templates.json", "w") as f:
            json.dump(unique_templates, f, indent=2)
            
        print(f"Saved {len(unique_templates)} unique_templates to data_templates.json")
        
if __name__ == "__main__":
    gen = TemplateGenerator()
    gen.generate_templates()