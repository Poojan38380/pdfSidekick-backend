import os
from typing import List, Dict, Any
from langchain_huggingface import HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from utils.colorLogger import print_info, print_error
from dotenv import load_dotenv
from huggingface_hub import InferenceClient


load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(
    provider="hf-inference",
    api_key=HUGGINGFACE_API_TOKEN,
)

# Updated default model to a more robust text generation model
DEFAULT_LLM_MODEL = "bigscience/bloom-560m"


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
            # Validate API token
            hf_api_token = HUGGINGFACE_API_TOKEN
            if not hf_api_token:
                raise ValueError("Missing Hugging Face API token")

            # More robust model initialization with additional parameters
            self.model = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",  # Changed from text2text-generation
                huggingfacehub_api_token=hf_api_token,
                model_kwargs={},
                do_sample=True,
                temperature=0.7,
                max_new_tokens=250,
            )
            print_info(f"Initialized LLM client with model: {model_name}")
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

            # More robust response handling
            response = client.chat.completions.create(
                model="Qwen/Qwen3-235B-A22B",
                messages=[{"role": "user", "content": prompt}],
            )

            return response.choices[0].message.content

        except Exception as e:
            print_error(f"Error generating answer (in llm_client.py): {str(e)}")
            # More comprehensive error logging
            import traceback

            print_error(f"Detailed Traceback: {traceback.format_exc()}")
            return "I apologize, but I encountered an error while trying to answer your question. Please try again."
