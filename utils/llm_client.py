import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from utils.colorLogger import print_info, print_error
from dotenv import load_dotenv

load_dotenv()

# Default LLM model - using a free model with Hugging Face Inference API
DEFAULT_LLM_MODEL = "google/flan-t5-small"


class LLMClient:
    """
    Client for handling LLM interactions using LangChain
    """

    def __init__(self, model_name: str = DEFAULT_LLM_MODEL):
        """
        Initialize the LLM client
        Args:
            model_name: The name of the LLM model to use
        """
        try:
            # Use Hugging Face's hosted inference API - no local GPU required
            self.model = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text2text-generation",
                max_length=512,
            )
            print_info(f"Initialized LLM client with model: {model_name}")
        except Exception as e:
            print_error(f"Error initializing LLM client: {e}")
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

            # Generate the answer
            response = self.model.invoke(prompt)
            return response.strip()

        except Exception as e:
            print_error(f"Error generating answer (in llm_client.py): {e}")
            return "I apologize, but I encountered an error while trying to answer your question. Please try again."
