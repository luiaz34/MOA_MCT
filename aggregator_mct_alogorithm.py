import os
import anthropic
import re
import math
import numpy as np
import random

def aggregate_evaluation_mct(prompt: str) -> str:
    """Aggregate evaluations using Claude 3.5 Sonnet."""

    anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    response = anthropic_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        temperature=0.2,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.content[0].text

def get_critique(question, draft_answer):
    prompt = (
        f"Question: {question}\n"
        f"Draft Answer: {draft_answer}\n"
        "Please critique the draft answer. "
        "Do a careful assessment of whether the answer is correct or not, and why. "
        "Consider multiple ways of verifying the correctness of the answer. "
        "Point out every flaw and hold the draft answer to a high standard. "
        "Provide specific recommendations to improve the answer. "
        "Think step by step. "
        "Do not provide a revised answer."
    )
    return aggregate_evaluation_mct(prompt)


def improve_answer(question, draft_answer, critique):
    prompt = (
        f"Question: {question}\n"
        f"Draft Answer: {draft_answer}\n"
        f"Critique: {critique}\n\n"
        "Please improve the draft answer based on the critique. Follow this format:\n"
        "Reasoning Process: <step-by-step reasoning process>\n"
        "Verification: <verification of the facts>\n"
        "Final Answer: <the improved and verified answer>\n"
    )
    return aggregate_evaluation_mct(prompt)


def rate_answer(question, answer):
    prompt = (
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        "As an expert on this topic, please provide a detailed critique of the answer, pointing out every flaw. "
        "Provide only a critique, not a suggested answer. "
        "Then, rate the answer on a scale of 0 to 100. "
        "The response should be in the following format:\n"
        "Critique: <detailed critique>\n"
        "Rating: <rating>"
    )
    rating_response = aggregate_evaluation_mct(prompt)
    try:
        match = re.search(r'Rating:\s*(\d+)', rating_response)
        rating = float(match.group(1)) / 100 if match else 0.0
    except Exception as e:
        print(f"Error extracting rating: {e}\nRating response was: {rating_response}")
        rating = 0.0
    return min(rating, 0.95)


class Node:
    def __init__(self, question, answer, parent=None, max_children=3):
        self.question = question
        self.answer = answer
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        self.max_children = max_children

    def is_fully_expanded(self):
        return len(self.children) >= self.max_children

    def best_child(self, exploration_weight=1.41):
        choices_weights = [
            (child.value / child.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits)
            if child.visits > 0 else float('inf')
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def most_visited_child(self):
        return max(self.children, key=lambda child: child.visits)

    def add_child(self, child_node):
        self.children.append(child_node)


class AggregatorMCT:

    def __init__(self, question, seed_answers=["I'm not sure", "I dont know", "I can't say it"], iterations=2, max_children=3):
        
        self.question = question
        self.seed_answers = seed_answers
        self.iterations = iterations
        self.max_children = max_children
        self.root = Node(question, random.choice(seed_answers), max_children=max_children)

    def search(self):
        for i in range(self.iterations):
            node = self.select(self.root)
            if not node.is_fully_expanded():
                node = self.expand(node)
            reward = self.simulate(node)
            self.backpropagate(node, reward)
        return self.root.most_visited_child().answer

    def select(self, node):
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        return node

    def expand(self, node):
        for _ in range(self.max_children - len(node.children)):
            child_node = Node(self.question, node.answer, parent=node, max_children=self.max_children)
            node.add_child(child_node)
            critique = get_critique(self.question, child_node.answer)
            child_node.answer = improve_answer(self.question, child_node.answer, critique)
        return random.choice(node.children)

    def simulate(self, node):
        return rate_answer(self.question, node.answer)

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent