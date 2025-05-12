import os
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langchain.prompts import PromptTemplate
from utils.colorLogger import print_info, print_error
from dotenv import load_dotenv

load_dotenv()

# Updated to a well-supported model
DEFAULT_LLM_MODEL = "google/flan-t5-base"


class LLMClient:
    """
    Client for handling LLM interactions using HuggingFace Transformers directly
    """

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the LLM client
        Args:
            model_name: The name of the LLM model to use
        """
        try:
            # Initialize tokenizer and model directly
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

            # Use CUDA if available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = self.model.to(self.device)

            print_info(
                f"Initialized LLM client with model: {model_name} on {self.device}"
            )
        except Exception as e:
            print_error(f"Error initializing LLM client: {str(e)}")
            raise

    async def generate_answer(
        self, question: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate an answer based on the question and context chunks
        Args:
            question: The user's question
            context_chunks: List of relevant context chunks from the PDF
        Returns:
            Generated answer
        """
        try:
            # Prepare the context from chunks
            context_parts = []
            for chunk in context_chunks:
                page_info = f"Page {chunk.get('page_number', 'Unknown')}"
                if chunk.get("title"):
                    page_info += f" of document: {chunk.get('title')}"

                context_parts.append(f"{page_info}:\n{chunk.get('content', '')}")

            context = "\n\n".join(context_parts)

            # Create the prompt template
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are a helpful AI assistant that answers questions based on the provided context from a PDF document.
                Use only the information from the context to answer the question. If the answer cannot be found in the context, say so.
                
                Be direct, concise and informative in your answers. If the context contains multiple pieces of relevant information, 
                organize your answer to present the most relevant information first.
                
                Context:
                {context}
                
                Question: {question}
                
                Answer:""",
            )

            # Format the prompt
            prompt = prompt_template.format(context=context, question=question)

            # Generate with transformers directly
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

            # Generate a response with safe parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=250,
                    temperature=0.7,
                    num_return_sequences=1,
                )

            # Decode the response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Return the response
            return (
                response.strip()
                if response
                else "I couldn't generate an answer based on the provided context."
            )

        except Exception as e:
            print_error(f"Error generating answer (in llm_client.py): {str(e)}")
            # More comprehensive error logging
            import traceback

            print_error(f"Detailed Traceback: {traceback.format_exc()}")
            return "I apologize, but I encountered an error while trying to answer your question. Please try again."
