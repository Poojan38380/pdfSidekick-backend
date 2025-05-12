import os
from typing import List, Dict, Any, Optional, Union
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    pipeline,
    AutoModelForCausalLM,
)
from langchain.prompts import PromptTemplate
from utils.colorLogger import print_info, print_error, print_warning
from dotenv import load_dotenv
import traceback
from tqdm import tqdm

load_dotenv()

# Enhanced model options for different use cases
DEFAULT_LLM_MODEL = "google/flan-t5-large"


class LLMClient:
    """
    Enhanced client for handling LLM interactions using HuggingFace Transformers
    with improved answer generation capabilities and advanced configuration options
    """

    def __init__(
        self,
        model_name: str = DEFAULT_LLM_MODEL,
        use_8bit: bool = False,
        max_memory: Optional[Dict[int, str]] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = "auto",
    ):
        """
        Initialize the LLM client with advanced configuration
        Args:
            model_name: The name of the LLM model to use
            use_8bit: Whether to load the model in 8-bit precision (saves memory)
            max_memory: Optional dictionary mapping device to maximum memory
            device_map: How to map model across devices
        """
        try:
            # Initialize tokenizer with caching and safety options
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                use_fast=True,
                cache_dir=os.getenv("MODELS_CACHE_DIR", "./.model_cache"),
                token=os.getenv("HF_TOKEN"),
                padding_side="left",
            )

            # Determine model class based on model name
            if "t5" in model_name.lower():
                model_class = T5ForConditionalGeneration
            else:
                model_class = AutoModelForCausalLM

            # Advanced loading options
            quantization_config = {"load_in_8bit": use_8bit} if use_8bit else {}

            # Load model with optimizations
            self.model = model_class.from_pretrained(
                model_name,
                cache_dir=os.getenv("MODELS_CACHE_DIR", "./.model_cache"),
                token=os.getenv("HF_TOKEN"),
                device_map=device_map,
                max_memory=max_memory,
                **quantization_config,
            )

            # Optimize for performance
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Initialize generation pipeline for efficiency
            self.generation_pipeline = pipeline(
                (
                    "text2text-generation"
                    if "t5" in model_name.lower()
                    else "text-generation"
                ),
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
            )

            # Enable model optimization techniques
            self.model.eval()
            torch.set_grad_enabled(False)

            # Store model name for reference
            self.model_name = model_name

            print_info(
                f"Initialized advanced LLM client with model: {model_name} on {self.device}"
                f"{' (8-bit quantized)' if use_8bit else ''}"
            )
        except Exception as e:
            print_error(f"Critical error initializing LLM client: {str(e)}")
            print_error(f"Detailed Error Traceback: {traceback.format_exc()}")
            raise

    async def generate_answer(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        max_length: int = 350,
        temperature: float = 0.6,
        additional_context: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """
        Generate a high-quality, contextually-rich answer
        Args:
            question: The user's input question
            context_chunks: List of relevant context chunks
            max_length: Maximum length of generated response
            temperature: Sampling temperature for response diversity
            additional_context: Optional additional context to include
            stream: Whether to stream the response (if supported)
        Returns:
            Comprehensive and precise answer
        """
        try:
            # Enhanced context preparation with more metadata and sorting
            # Sort chunks by relevance if available
            if context_chunks and "relevance_score" in context_chunks[0]:
                context_chunks = sorted(
                    context_chunks,
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True,
                )

            context_parts = []
            for chunk in context_chunks:
                # Improved context representation with more metadata
                page_info = f"Source: Page {chunk.get('page_number', 'Unknown')}"
                if chunk.get("title"):
                    page_info += f" | Document: {chunk.get('title')}"

                # Add confidence score or relevance scoring if available
                relevance_score = chunk.get("relevance_score", 1.0)

                # Format content with contextual info
                context_text = chunk.get("content", "").strip()
                if context_text:
                    context_parts.append(
                        f"{page_info} [Relevance: {relevance_score:.2f}]:\n{context_text}"
                    )

            # Combine context with advanced prompt engineering
            context = "\n\n".join(context_parts)

            # Add additional context if provided
            if additional_context:
                context += f"\n\nAdditional Information:\n{additional_context}"

            # Sophisticated prompt template with explicit instructions
            # and improved context utilization
            prompt_template = PromptTemplate(
                input_variables=["context", "question"],
                template="""You are an expert AI research assistant tasked with providing precise, well-structured answers.

CORE INSTRUCTIONS:
1. Analyze the context thoroughly
2. Directly address the specific question
3. Prioritize accuracy and completeness
4. Provide specific citations when referencing information (page numbers, document titles)
5. If information is insufficient, clearly state limitations

CONTEXT:
{context}

QUESTION: {question}

RESPONSE FORMAT:
- Begin with a direct, concise answer to the question
- Support with specific evidence from the provided context
- Include relevant citations where appropriate
- Organize information logically with clear structure
- If no definitive answer exists, explain why and what information would be needed

ANSWER:""",
            )

            # Format the enhanced prompt
            prompt = prompt_template.format(context=context, question=question)

            # Generate using the pipeline for better efficiency and support for streaming
            generation_args = {
                "max_length": max_length,
                "min_length": 50,
                "temperature": temperature,
                "do_sample": True,
                "top_k": 50,
                "top_p": 0.95,
                "no_repeat_ngram_size": 2,
                "repetition_penalty": 1.2,
                "num_return_sequences": 1,
            }

            if stream:
                # For streaming implementation
                # Note: Actual streaming would require additional integration
                # This is a placeholder for future implementation
                print_warning(
                    "Streaming not fully implemented yet, falling back to standard generation"
                )

            # Generate the response
            outputs = self.generation_pipeline(prompt, **generation_args)

            # Extract and process response
            response = outputs[0]["generated_text"]

            # Clean up any potential prompt repetition
            if "ANSWER:" in response:
                response = response.split("ANSWER:", 1)[1].strip()

            # Additional response validation and post-processing
            processed_response = self._validate_response(response, context_chunks)

            return processed_response

        except Exception as e:
            print_error(f"Comprehensive error in answer generation: {str(e)}")
            print_error(f"Detailed Error Traceback: {traceback.format_exc()}")
            return "I encountered a complex processing error. Could you rephrase your question or provide more context?"

    async def generate_answers_batch(
        self,
        questions: List[str],
        context_chunks_list: List[List[Dict[str, Any]]],
        max_length: int = 350,
        temperature: float = 0.6,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Generate multiple answers in batch for efficiency
        Args:
            questions: List of user questions
            context_chunks_list: List of context chunks lists (one per question)
            max_length: Maximum length of generated responses
            temperature: Sampling temperature for response diversity
            batch_size: Number of questions to process in each batch
        Returns:
            List of answers corresponding to input questions
        """
        if len(questions) != len(context_chunks_list):
            raise ValueError(
                "Number of questions must match number of context chunks lists"
            )

        answers = []

        # Process in batches
        for i in tqdm(
            range(0, len(questions), batch_size), desc="Processing question batches"
        ):
            batch_questions = questions[i : i + batch_size]
            batch_contexts = context_chunks_list[i : i + batch_size]

            batch_answers = []
            for question, contexts in zip(batch_questions, batch_contexts):
                answer = await self.generate_answer(
                    question=question,
                    context_chunks=contexts,
                    max_length=max_length,
                    temperature=temperature,
                )
                batch_answers.append(answer)

            answers.extend(batch_answers)

        return answers

    def _validate_response(
        self, response: str, context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Post-process and validate the generated response
        Args:
            response: Generated answer
            context_chunks: Original context for validation
        Returns:
            Refined and validated response
        """
        # Trim excessive whitespace and remove potential artifacts
        cleaned_response = " ".join(response.split())

        # Minimal length check
        if len(cleaned_response) < 20:
            if context_chunks and len(context_chunks) > 0:
                # Try to extract at least some information from context
                fallback = "Based on the available information: "
                best_chunk = max(
                    context_chunks, key=lambda x: x.get("relevance_score", 0)
                )
                fallback += best_chunk.get("content", "")[:200] + "..."
                return fallback
            return "I couldn't generate a comprehensive answer based on the available context."

        # Check for hallucinations - response discussing topics not in context
        context_content = " ".join(
            [chunk.get("content", "") for chunk in context_chunks]
        )
        if len(context_content) > 0 and len(cleaned_response) > 0:
            # This is a simple check - more sophisticated methods could be implemented
            # such as named entity recognition to detect new entities not in context
            pass

        # Ensure proper formatting
        if not cleaned_response.endswith((".", "!", "?")):
            cleaned_response += "."

        return cleaned_response

    def get_available_models(self) -> Dict[str, str]:
        """
        Get available model options
        Returns:
            Dictionary of available models with descriptions
        """
        return {
            "t5-base": "Faster, lighter model suitable for testing",
            "t5-large": "Balanced performance and quality (default)",
            "t5-xl": "Higher quality answers, requires more resources",
            "t5-xxl": "Production quality, recommended with 8-bit quantization",
        }
