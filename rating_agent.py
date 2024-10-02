import os
import anthropic
import asyncio
from typing import List, Tuple

class RatingAgent:
    @staticmethod
    async def claudeRatingAgent(userQuery: str, proposersResponse: List[Tuple[str, str]], aggregatorsResponse: List[Tuple[str, str]]):
        try:
            anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
            
            system_prompt = """
            You are a rating agent tasked with evaluating responses from multiple Proposers and Aggregators based on a user query. 
            Your job is to rate each response on a scale from 0.1 to 10.0, where 0.1 is the lowest possible score and 10.0 is the highest possible score.
            
            For each response:
            1. Carefully read and analyze the response in relation to the user query.
            2. Consider factors such as relevance, accuracy, completeness, clarity, and helpfulness.
            3. Determine a score between 0.1 and 10.0 that best represents the quality of the response.
            4. Provide a brief justification for your rating, explaining your reasoning.

            Present your evaluation for each response in the following format:

            <evaluation>
            <model>[Model Name]</model>
            <justification>
            [Your justification for the response rating]
            </justification>
            <score>[Your score for the response]</score>
            </evaluation>

            Remember to be objective and consistent in your evaluations, focusing on how well each response addresses the user query.
            """

            user_message = f"""
            User Query: {userQuery}

            Proposer Responses:
            {RatingAgent._format_responses(proposersResponse)}

            Aggregator Responses:
            {RatingAgent._format_responses(aggregatorsResponse)}

            Please evaluate each response individually.
            """

            response = anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=4096,
                temperature=0.1,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            await asyncio.sleep(1)
            return ".........................................RATING AGENT..................................\n" + response.content[0].text + "\n........................................................................\n"
        except Exception as e:
            print(f"An unexpected error when calling claude-Rating-Agent occurred: {e}")
        return None

    @staticmethod
    def _format_responses(responses: List[Tuple[str, str]]) -> str:
        formattedResponse = ""
        for model, response in responses:
            formattedResponse += f"{model}:\n{response}\n\n"
        return formattedResponse
