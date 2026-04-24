import os
from groq import Groq
from dotenv import load_dotenv
import json

load_dotenv()

class RAGEngine:
    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.3-70b-versatile"  # Successor to Llama 3 70B

    def generate_answer(self, query, context_chunks):
        """
        Generates a grounded answer with citations.
        """
        # Format context for the prompt
        formatted_context = ""
        for i, chunk in enumerate(context_chunks):
            meta = chunk['metadata']
            formatted_context += f"--- CONTEXT CHUNK {i+1} ---\n"
            formatted_context += f"Source: {meta['filename']}\n"
            formatted_context += f"Location: {meta['location']}\n"
            formatted_context += f"Content: {chunk['content']}\n\n"

        system_prompt = """
        You are a Knowledge Assistant. Use the provided Context Chunks to answer the User Query.
        
        STRICT RULES:
        1. Answer ONLY using the provided Context.
        2. If the answer is not in the Context, return: {"error": "Information not found in selected documents."}
        3. Do NOT use your own knowledge.
        4. Every claim must be backed by a quote from the context.
        5. Return your response ONLY in the following JSON format:
        {
            "answer": "A concise 1-3 sentence answer.",
            "citation": {
                "quote": "The exact direct quote from the source.",
                "source": "Filename",
                "location": "Page X or Tab/Row"
            }
        }
        6. No greetings, no introductions, no conversational filler.
        """

        user_prompt = f"Context:\n{formatted_context}\n\nUser Query: {query}"

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.model,
                temperature=0.0,  # Zero temperature for deterministic grounded answers
                max_tokens=300,   # Token optimization
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            return {"error": f"LLM Error: {str(e)}"}
