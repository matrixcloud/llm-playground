import json
from libs.client import AiClient

class IterativeRefinement:
    def __init__(self, ai: AiClient, initial_prompt: str, max_iter = 3):
        self.ai = ai
        self.initial_prompt = initial_prompt
        self.max_iter = max_iter

    def perform(self):
        """
        A generator that yields a prompt for the user to answer.
        """
        print(f"Your initial prompt: {self.initial_prompt}")
        prompt_initialization = """
        Your name is now 'Promptor' and that is how I will address you from now on.
        Promptor and GPT are separate and distinct entities.
        You are an expert in prompt engineering and large language models.
        A good prompt should assign one or many roles to GPT, define a clear context
        and task, and clarify expected output. You know and use many prompt
        techniques such as Few-Shot Learning, Prompt Chaining, Shadow Prompting, ...
        I want you to be my personal prompt creator expert.
        You, Promptor, are responsible for creating good prompts for GPT.
        """
        current_prompt = self.initial_prompt
        questions_answers = ""
        for i in range(self.max_iter):
            print(f"Loop {i+1}")
            reviews = self.reviewer(prompt_initialization, current_prompt)
            questions_answers = self.questioner(
                prompt_initialization, current_prompt, reviews, questions_answers)
            current_prompt = self.prompt_maker(
                prompt_initialization, current_prompt, reviews, questions_answers)

            print(f"\nNew current prompt: {current_prompt}\n\n")
            keep = input(f"Do you want to keep this prompt (y/n)? ")
            if keep == 'y':
                break

        return current_prompt

    def reviewer(self, prompt_initialization, current_prompt):
        prompt_reviewer = prompt_initialization + "\n\n"
        prompt_reviewer += f"This is my prompt: {current_prompt}\n\n"
        prompt_reviewer += """
        Task: Provide a detailed, rigorous critique of my prompt.
        To do this, first start by giving my prompt a score from 0 to 5
        (0 for poor, 5 for very optimal), and then write a short paragraph
        detailing improvements that would make my prompt a perfect prompt
        with a score of 5."""

        reviews = self.ai.complete(prompt_reviewer)

        return reviews

    def questioner(self, prompt_initialization, current_prompt, reviews, questions_answers):
        prompt_questioner = prompt_initialization + "\n\n"
        prompt_questioner += f"This is my prompt: {current_prompt}\n\n"
        prompt_questioner += f"A critical review of my prompt:{reviews}\n\n"
        prompt_questioner += """Task: Compile a list of maximum 4 short questions
        whose answers are indispensable for improving my prompt (also give examples
        of answers in baskets.).
        Output format: In JSON format. The output must be accepted by json.loads.
        The json format should be similar to:
        {'Questions': ['Question 1','Question 2','Question 3','Question 4']}"""

        questions_json = self.ai.complete(
            prompt_questioner,
            response_format={"type": "json_object"}
        )

        try:
            questions = json.loads(questions_json).get('Questions', [])
        except json.JSONDecodeError:
            print("Failed to decode questions from the model's response.")
            questions = []

        for i, question in enumerate(questions, start=1):
            answer = input(f"Question {i}: {question} ")
            questions_answers = questions_answers + \
                f"Question: {question}\nAnswer:{answer}\n\n"

        return questions_answers

    def prompt_maker(self, prompt_initialization, current_prompt, reviews, questions_answers):
        prompt = prompt_initialization + "\n\n"
        prompt += f"This is my current prompt: {current_prompt}\n\n"
        prompt += f"This is critical review of my current prompt:{reviews}\n\n"
        prompt += f"Some questions and answers for improving my current prompt:{questions_answers}\n\n"
        prompt += """Task: With all of this information, use all of your prompt
        engineering expertise to rewrite my current prompt in the best possible
        way to create a perfect prompt for GPT with a score of 5. All the
        information contained in the questions and answers must be included in
        the new prompt. Start the prompt by assigning one or many roles to GPT,
        defining the context, and the task.
        Output: It's very important that you only return the new prompt for GPT
        that you've created, and nothing else."""
        new_prompt = self.ai.complete(prompt)
        return (new_prompt)

def main():
    refinement = IterativeRefinement(
        ai=AiClient(),
        initial_prompt="Give me a suggestion for the main course for today's lunch."
    )
    res = refinement.perform()
    print(res)

if __name__ == '__main__':
    main()
