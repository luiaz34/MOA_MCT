import os
import anthropic
import asyncio
import openai
from dotenv import load_dotenv
from typing import Tuple, List
import aggregator_mct_alogorithm

load_dotenv()

class Aggregator:
    
    async def claudeAggregator(proposerResult: list, userPrompt: str):
        try:
            anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            systemPrompt = f""" 
            You are an aggregator based on MOA (Mixture of Adapters) by Together AI. 
            Your task is to analyze responses from multiple proposers (large language models) to a given user query, compare them, and select the best answer or modify and give your best answer if it is necessary.

            Here is the user query you need to consider:
            <user_query>
            {userPrompt}
            </user_query>

            Now, here are the responses from various proposers:
            <proposer_responses>
            {proposerResult}
            </proposer_responses>

            To complete your task, follow these steps:

            1. Carefully read the user query and all proposer responses.

            2. Analyze each response for:
            - Relevance to the user query
            - Accuracy of information
            - Completeness of the answer
            - Clarity and coherence

            3. Compare the responses to identify:
            - Common points across multiple responses
            - Unique insights provided by individual proposers
            - Any contradictions or inconsistencies between responses

            4. Select the best answer based on the following criteria:
            - Most directly addresses the user query
            - Provides the most accurate and complete information
            - Offers clear and well-structured explanation
            - Incorporates valuable insights from multiple proposers if applicable

            5. If no single response fully satisfies the criteria, synthesize a new answer using the best elements from multiple responses.

            6. Provide your final answer in the following format:
            <best_answer>
            [Insert the selected or synthesized best answer here]
            </best_answer>

            <reasoning>
            [Explain why this answer was chosen or how it was synthesized, referencing specific strengths of the selected response(s) and how they relate to the user query]
            </reasoning>

            Remember, your goal is to provide the most helpful and accurate answer to the user query by leveraging the collective knowledge of the proposers. Ensure that your final answer is coherent, well-structured, and directly addresses the user's question.
            """
            response = anthropic_client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=8000,
                temperature=0.2,
                system=systemPrompt,
                messages=[
                    {"role": "user", "content": userPrompt}
                ]
            )
            await asyncio.sleep(1)
            return response.content[0].text
        except anthropic.APIError as e:
            print(f"An API error when calling Anthropic Aggregator occurred: {e}")
        except Exception as e:
            print(f"An unexpected error when calling Anthropic Aggregator occurred: {e}")
        return None
    
    async def GPTAggregator(proposerResult: list, userPrompt: str):
        try:
            openai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
            systemPrompt = f""" 
            You are an aggregator based on MOA (Mixture of Adapters) by Together AI. 
            Your task is to analyze responses from multiple proposers (large language models) to a given user query, compare them, and select the best answer or modify and give your best answer if it is necessary.

            Here is the user query you need to consider:
            <user_query>
            {userPrompt}
            </user_query>

            Now, here are the responses from various proposers:
            <proposer_responses>
            {proposerResult}
            </proposer_responses>

            To complete your task, follow these steps:

            1. Carefully read the user query and all proposer responses.

            2. Analyze each response for:
            - Relevance to the user query
            - Accuracy of information
            - Completeness of the answer
            - Clarity and coherence

            3. Compare the responses to identify:
            - Common points across multiple responses
            - Unique insights provided by individual proposers
            - Any contradictions or inconsistencies between responses

            4. Select the best answer based on the following criteria:
            - Most directly addresses the user query
            - Provides the most accurate and complete information
            - Offers clear and well-structured explanation
            - Incorporates valuable insights from multiple proposers if applicable

            5. If no single response fully satisfies the criteria, synthesize a new answer using the best elements from multiple responses.

            6. Provide your final answer in the following format:
            <best_answer>
            [Insert the selected or synthesized best answer here]
            </best_answer>

            <reasoning>
            [Explain why this answer was chosen or how it was synthesized, referencing specific strengths of the selected response(s) and how they relate to the user query]
            </reasoning>

            Remember, your goal is to provide the most helpful and accurate answer to the user query by leveraging the collective knowledge of the proposers. Ensure that your final answer is coherent, well-structured, and directly addresses the user's question.
            """
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                max_tokens=4096,
                temperature=0.2,
                messages=[
                    {"role": "system", "content": systemPrompt},
                    {"role": "user", "content": userPrompt}
                ]
            )
            await asyncio.sleep(1)
            return response.choices[0].message.content
        except openai.APIError as e:
            print(f"An API error when calling OpenAI Aggregator occurred: {e}")
        except Exception as e:
            print(f"An unexpected error when calling OpenAI Aggregator occurred: {e}")
        return None
    
    async def processMCTAggregator(proposerResult: List[Tuple[str, str]], max_children: int, iteration: int, userPrompt: str) -> str:
        aggregator_mct =  aggregator_mct_alogorithm.AggregatorMCT(question=userPrompt, seed_answers=proposerResult, iterations=iteration, max_children=max_children)
        final_evaluation = aggregator_mct.search()
        return final_evaluation
