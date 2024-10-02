from typing import List, Tuple
import asyncio
import openai
import os

class SelectorAgent:

    @staticmethod
    async def answerSelector(proposerResponse: List[Tuple[str, str]], aggregatorResponse: List[Tuple[str, str]]):
        try:
            openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            
            system_prompt = """
            You're a selector agent tasked with identifying the best response from proposer and aggregator outputs. Your task:
            1. Analyze responses for accuracy, completeness, clarity, and relevance.
            2. Identify the single best answer.
            3. Select the best answer WITHOUT modifying it in any way.
            4. Briefly explain why it's the best, highlighting key strengths.

            Format your response as follows:
            <selected_model>[Model Name]</selected_model>
            <best_answer>
            [Exact text of the best answer, copied without any changes]
            </best_answer>
            <reasoning>
            [Short explanation of why this is the best, key strengths]
            </reasoning>

            Important: Do not alter, rephrase, or improve the selected answer. Copy it exactly as provided.
            Be concise in your reasoning and focus on the most important aspects.
            """
            user_message = f"""
            User Query: Explain Bayes' theorem
            Candidate Responses:
            {SelectorAgent._format_responses(proposerResponse + aggregatorResponse)}
            Select the best answer based on the criteria above, without modifying it.
            """
            response = openai_client.chat.completions.create(
                model="gpt-4",
                max_tokens=2048,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ]
            )
            return response.choices[0].message.content.strip()
        except openai.APIError as e:
            print(f"An API error when calling OpenAI Selector occurred: {e}")
        except Exception as e:
            print(f"An unexpected error when calling OpenAI Selector occurred: {e}")
        return None

    @staticmethod
    def _format_responses(responses: List[Tuple[str, str]]) -> str:
        formatted_responses = ""
        for model, response in responses:
            formatted_responses += f"<model>{model}</model>\n<response>{response}</response>\n\n"
        return formatted_responses.strip()