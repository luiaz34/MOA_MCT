import proposer
import dto
import aggregator
import rating_agent
import selector_agent
import asyncio

userPrompt = "explain the Bayes' theorem"
temp = 0.3
topP = 1

# Non-zero
proposerCount = 1
aggregatorCount = 0
ratingAgent = False
selectorAgent = False

useMCT = False
maxChildren = 3
iteration = 2
# topK = 1,

TotalNumberOfModel = ['llama', 'gemma', 'mixtral']


async def executeProgram(dataDTO: dto.DataDTO):
    if dataDTO.proposer_count > len(TotalNumberOfModel):
        return f"There are only {len(TotalNumberOfModel)} models available"

    proposerInit = proposer.Proposer(dataDTO.user_prompt, dataDTO.temp, dataDTO.top_p)
    responses = []

    # Proposer selection logic
    if dataDTO.proposer_count == 1:
        llamaResponse = await proposerInit.llama_proposer()
        responses.append(("------------------------------------- LLAMA PROPOSER RESPONSE --------------------------------------",llamaResponse))

    elif dataDTO.proposer_count == 2:
        llamaResponse, gemmaResponse = await asyncio.gather(proposerInit.llama_proposer(), proposerInit.gemma_proposer())
        responses.extend([
            (
            "-------------------------------------LLAMA PROPOSER RESPONSE--------------------------------------",
            llamaResponse),
            (
            "-------------------------------------GEMMA PROPOSER RESPONSE--------------------------------------",
            gemmaResponse)
        ])

        
    elif dataDTO.proposer_count == 3:
        llamaResponse, gemmaResponse, mixtralResponse = await asyncio.gather(
            proposerInit.llama_proposer(),
            proposerInit.gemma_proposer(),
            proposerInit.mixtral_proposer()
        )
        responses.extend([
            (
            "------------------------------------------LLAMA PROPOSER RESPONSE-------------------------------------------",
            llamaResponse
            ),
            (
            "------------------------------------------GEMMA PROPOSER RESPONSE-------------------------------------------",
            gemmaResponse
            ),
            (
            "------------------------------------------MIXTRAL PROPOSER RESPONSE-------------------------------------------",
            mixtralResponse
            )
        ])

    else:
        return "Invalid proposer count"

    # If there are no aggregators and no rating/selector agents, return the proposer responses
    if dataDTO.aggregator_count == 0 and not dataDTO.rating_agent and not dataDTO.selector_agent:
        return responses

    # Aggregator logic
    aggregator_results = []
    for i in range(dataDTO.aggregator_count):
        if i == 0:
            if dataDTO.use_mct:
                result = await aggregator.Aggregator.processMCTAggregator([r[1] for r in responses],
                                                                          dataDTO.max_children, dataDTO.iteration,
                                                                          dataDTO.user_prompt)
                print(
                    "----------------------------------------- USE MCT CLAUDE AGGREGATOR RESPONSE -------------------------------------------")
                aggregator_results.append((
                                          "----------------------------------------- USE MCT CLAUDE AGGREGATOR RESPONSE -------------------------------------------",
                                          result))

            else:
                result = await aggregator.Aggregator.claudeAggregator([r[1] for r in responses], dataDTO.user_prompt)
                aggregator_results.append((
                                          "----------------------------------------- CLAUDE AGGREGATOR RESPONSE -------------------------------------------",
                                          result))

        else:
            # Subsequent aggregators use GPT-4
            all_inputs = [r[1] for r in responses] + [r[1] for r in aggregator_results]
            result = await aggregator.Aggregator.GPTAggregator(all_inputs, dataDTO.user_prompt)
            aggregator_results.append((
                                      "------------------------------------------GPT AGGREGATOR RESPONSE-------------------------------------------",
                                      result))

    # Combine all responses and aggregations
    all_results = responses + aggregator_results
    for model, response in all_results:
        print(f"{model}:\n{response}\n")

    if dataDTO.rating_agent and not dataDTO.selector_agent:
        ratingAgentResponse = await rating_agent.RatingAgent.claudeRatingAgent(dataDTO.user_prompt, responses,
                                                                               aggregator_results)
        print(
            "------------------------------------------ RATING AGENT RESPONSE ------------------------------------------")
        print(ratingAgentResponse)
        return ratingAgentResponse
    elif dataDTO.selector_agent and not dataDTO.rating_agent:
        selectorAgentResponse = await selector_agent.SelectorAgent.answerSelector(responses, aggregator_results)
        print(
            "------------------------------------------ SELECTOR AGENT RESPONSE ------------------------------------------")
        print(selectorAgentResponse)
        return selectorAgentResponse
    elif dataDTO.rating_agent and dataDTO.selector_agent:
        ratingAgentResponse = await rating_agent.RatingAgent.claudeRatingAgent(dataDTO.user_prompt, responses,
                                                                               aggregator_results)
        selectorAgentResponse = await selector_agent.SelectorAgent.answerSelector(responses, aggregator_results)

        print(
            "------------------------------------------ START RATING AGENT RESPONSE ------------------------------------------")
        print(ratingAgentResponse)

        print(
            "------------------------------------------ START SELECTOR AGENT RESPONSE ------------------------------------------")
        print(selectorAgentResponse)

        return selectorAgentResponse
    else:
        print("---------- Aggregator Results: ----------")
        for model, response in aggregator_results:
            print(f"{model}:\n{response}\n")
        return aggregator_results


# App Entry Point
async def main():
    answer = await executeProgram(
        dataDTO=dto.DataDTO(user_prompt=userPrompt, top_p=topP, proposer_count=proposerCount, temp=temp,
                            aggregator_count=aggregatorCount, rating_agent=ratingAgent, selector_agent=selectorAgent,
                            use_mct=useMCT, iteration=iteration, max_children=maxChildren))
    print(answer)

if __name__ == "__main__":
    asyncio.run(main())
