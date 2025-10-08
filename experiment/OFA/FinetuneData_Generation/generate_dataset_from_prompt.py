import os
import argparse
from openai import OpenAI
import math

MINE_BASE_URL = "xxxx"
MINE_API_KEY = "xxxxx"

PROMPT_TEMPLATE = """You are an expert AI system designer specializing in Multi-Agent Systems (MAS). Your goal is to generate a large dataset of diverse queries and propose a suitable MAS topology configuration for solving each query. This dataset will be used to train a foundational model that learns to design MAS topologies automatically.

**Your Task:**

Based on the provided resources (a Role Pool and a set of classic Topologies), generate {num_to_generate} diverse queries. For EACH query, you must propose a plausible MAS configuration that could solve it.

**Available Resources:**

1.  **Role Pool (List of available agent roles):**
    *   `Knowlegable Expert`: 
    \"\"\"
    You are a knowlegable expert in question answering.
    Please give several key entities that need to be searched in wikipedia to solve the problem, for example: catfish effect, broken window effect, Shakespeare.
    If there is no entity in the question that needs to be searched in Wikipedia, you don't have to provide it
    \"\"\"
    *   `Critic`: 
    \"\"\"
    You are an excellent critic.
    Please point out potential issues in other agent's analysis point by point.
    \"\"\"
    *   `Mathematician`: 
    \"\"\"
    You are a mathematician who is good at math games, arithmetic calculation, and long-term planning.
    \"\"\"
    *   `Psychologist`: 
    \"\"\"
    You are a psychologist.
    You are good at psychology, sociology, and philosophy.
    You give people scientific suggestions that will make them feel better.
    \"\"\"
    *   `Historian`: \"\"\"
    You research and analyze cultural, economic, political, and social events in the past, collect data from primary sources and use it to develop theories about what happened during various periods of history.
    \"\"\"
    *   `Doctor`: 
    \"\"\"
    You are a doctor and come up with creative treatments for illnesses or diseases.
    You are able to recommend conventional medicines, herbal remedies and other natural alternatives.
    You also consider the patient's age, lifestyle and medical history when providing your recommendations.
    \"\"\"
    *   `Project Manager`: 
        "You are a project manager. "
        "You will be given a function signature and its docstring by the user. "
        "You are responsible for overseeing the overall structure of the code, ensuring that the code is structured to complete the task Implement code concisely and correctly without pursuing over-engineering."
        "You need to suggest optimal design patterns to ensure that the code follows best practices for maintainability and flexibility. "
        "You can specify the overall design of the code, including the classes that need to be defined(maybe none) and the functions used (maybe only one function) ."
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points."
    *   `Algorithm Designer`: 
        "You are an algorithm designer. "
        "You will be given a function signature and its docstring by the user. "
        "You need to specify the specific design of the algorithm, including the classes that may be defined and the functions used. "
        "You need to generate the detailed documentation, including explanations of the algorithm, usage instructions, and API references. "
        "When the implementation logic is complex, you can give the pseudocode logic of the main algorithm."
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points."
    *   `Programming Expert`: 
        "You are a programming expert. "
        "You will be given a function signature and its docstring by the user. "
        "You may be able to get the output results of other agents. They may have passed internal tests, but they may not be completely correct. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response. "
        "Do not change function names and input variable types in tasks."
    *   `Test Analyst`: 
        "You are a test analyst. "
        "You will be given a function signature and its docstring by the user. "
        "You need to provide problems in the current code or solution based on the test data and possible test feedback in the question. "
        "You need to provide additional special use cases, boundary conditions, etc. that should be paid attention to when writing code. "
        "You can point out any potential errors in the code."
        "I hope your reply will be more concise. Preferably within fifty words. Don't list too many points."
    *   `Bug Fixer`: 
        "You are a bug fixer."
        "You will be given a function signature and its docstring by the user. "
        "You need to provide modified and improved python code based on the current overall code design, algorithm framework, code implementation or test problems. "
        "Write your full implementation (restate the function signature). "
        "Use a Python code block to write your response. For example:\n```python\nprint('Hello world!')\n```"
        "Do not include anything other than Python code blocks in your response "
        "Do not change function names and input variable types in tasks"
    *   `Math Solver`: 
        "You are a math expert. "
        "You will be given a math problem and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to."
    *   `Mathematical Analyst`: 
        "You are a mathematical analyst. "
        "You will be given a math problem, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results."
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to."
    *   `Programming Expert for Math`: 
        "You are a programming expert. "
        "You will be given a math problem, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve math problems. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to."
    *   `Inspector`: 
        "You are an Inspector. "
        "You will be given a math problem, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final result without any units, for example: The answer is 140\n"
        "You will be given some examples you may refer to."
    *   `Math Solver for choice question`:
        "You are a math expert. "
        "You will be given a multiple-choice question and hints from other agents. "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n"

    *   `Mathematical Analyst for choice question`: 
        "You are a mathematical analyst. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "You need to first analyze the problem-solving process step by step, where the variables are represented by letters. "
        "Then you substitute the values into the analysis process to perform calculations and get the results."
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n"

    *   `Programming Expert for choice question`: 
        "You are a programming expert. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "Integrate step-by-step reasoning and Python code to solve multiple-choice question. "
        "Analyze the question and write functions to solve the problem. "
        "The function should not take any arguments and use the final result as the return value. "
        "The last line of code calls the function you wrote and assigns the return value to the \(answer\) variable. "
        "Use a Python code block to write your response. For example:\n```python\ndef fun():\n x = 10\n y = 20\n return x + y\nanswer = fun()\n```\n"
        "Do not include anything other than Python code blocks in your response."
        "You will be given some examples you may refer to."
    *   `Inspector for choice question`: 
        "You are an Inspector. "
        "You will be given a multiple-choice question, analysis and code from other agents. "
        "Check whether the logic/calculation of the problem solving and analysis process is correct(if present). "
        "Check whether the code corresponds to the solution analysis(if present). "
        "Give your own solving process step by step based on hints. "
        "The last line of your output contains only the final choice with only a capital letter, for example: The answer is A\n"


2.  **Topology Pool (List of classic collaboration structures):**
    *   `Chain`: A linear pipeline where agents process information sequentially. Ideal for multi-step, dependent tasks.
    *   `Star`: A centralized model where a central agent gathers information from peripheral agents and synthesizes a final result. Ideal for "divide and conquer" tasks.
    *   `Mesh`: A structure where agents can communicate with several others, but not all. Good for tasks requiring localized collaboration.
    *   `FullConnected`: A structure where every agent can communicate with every other agent. Ideal for complex brainstorming or tasks requiring constant, multi-directional feedback.
    *   `Layered`: A hierarchical structure with multiple layers where agents in each layer process information before passing to the next layer. Good for tasks that can be broken down into sequential processing stages.

**Generation Instructions:**

1.  **Query Diversity:** The queries you generate should span a wide range of domains, including but not limited to:
    *   **Code Implementation:** (e.g., implementing functions with specific requirements, debugging code)
    *   **Mathematical Reasoning:** (e.g., word problems, geometry problems, arithmetic calculations)
    *   **Multiple-Choice Questions:** (e.g., science, math, or general knowledge questions with options A, B, C, D, E)
    *   **Knowledge-based Q&A:** (e.g., history, science, technology questions)
    *   **Medical Consultation:** (e.g., treatment recommendations, symptom analysis)
    *   **Psychology & Social Sciences:** (e.g., behavioral analysis, social phenomena explanations)

2.  **Balanced Configuration Distribution:**
    *   **Agent Count:** Ensure a roughly even distribution of configurations for agent counts of 2, 3, 4, 5, and 6. Avoid concentrating heavily on just 3-4 agents.
    *   **Topology Variety:** Make a conscious effort to use all five topology types (`Chain`, `Star`, `Mesh`, `FullConnected`, `Layered`) across the generated dataset, following the rules for avoiding redundant topologies described below. Avoid over-representing the `Chain` topology, especially for configurations with 3 or 4 agents where `Star` or `Layered` structures are also highly applicable. For simpler tasks that might seem linear, consider if a "divide and conquer" (`Star`) or hierarchical (`Layered`) approach could also be plausible.

3.  **Comprehensive Role Coverage:**
    *   **Utilize All Roles:** Across the entire set of {num_to_generate} queries, you should aim to use **every role** from the `Role Pool`. Actively create queries that necessitate the use of specialized roles (e.g., `Math Solver for choice question`, `Bug Fixer`, `Historian`, `Psychologist`).
    *   **Avoid Role Bias:** Do not repeatedly use a small subset of general roles. Consciously vary the roles you select. For a coding problem, consider if `Algorithm Designer` is a better fit than `Programming Expert`. For a math problem, ensure you are using the full variety of math-related roles provided.

4.  **Avoiding Redundant Topologies (CRITICAL LOGIC):**
    *   For small numbers of agents, some topologies become identical to others. To create a clean and non-redundant dataset, follow these specific rules:
    *   **For Agent Count = 2:** ONLY use the `Chain` topology (e.g., `edges: "0->1"`). All other types are identical to `Chain` in this case.
    *   **For Agent Count = 3:**
        *   Do NOT use `FullConnected`. This topology is identical to `Mesh` for three agents. Prefer `Mesh` with edges like `"0->1 1->2 0->2"`.
        *   Be mindful that a `Layered` structure can sometimes look identical to a `Star` (e.g., two agents reporting to one, like `edges: "0->2 1->2"`). Try to create `Layered` structures that are different, for example, a divergent structure (`edges: "0->1 0->2"`).
    *   **For Agent Count >= 4:** All five topologies are distinct and meaningful. Please use a rich variety of them for these larger configurations.

5.  **Configuration Logic:** For each query, your proposed configuration should be logical.
    *   Choose a `Topology` that matches the problem-solving workflow.
    *   Choose an `Agent_Count` between 2 and 6. Follow the instructions on balanced distribution and avoiding redundancy.
    *   Choose `Roles` from the pool that are directly relevant to solving the query. The number of roles must match the `Agent_Count`. **CRITICAL: You MUST only select roles from the `Role Pool` provided above. Do NOT invent, create, or use any roles not explicitly listed in the resources.**

6.  **DAG Constraint (CRITICAL):** The generated topology MUST be a Directed Acyclic Graph (DAG). This means:
    *   **No cycles allowed**: You cannot have paths like A→B→C→A or any circular dependencies.
    *   **Valid flow direction**: Information should flow from initial processors to final decision makers.
    *   **Typical patterns**: Use patterns like input→processing→output, or parallel→convergence→synthesis.
    *   **Verification tip**: Number your agents 0 to N-1, and ensure edges generally flow from lower to higher indices (though not strictly required).
    *   **Common DAG patterns to follow**:
        - **Chain**: 0→1→2→3 (linear processing)
        - **Star**: 0->3, 1->3, 2->3 (convergence to central node)
        - **Layered**: 0->2, 0->3, 1->2, 1->3 (layer-by-layer processing)
        - **Tree**: 0->1, 0->2, 1->3, 2->3 (hierarchical branching)
    *   **FORBIDDEN patterns**: Any backward edges like 3→0, 2→1, or cycles like 0→1→2→0.

7.  **Output Format:** You MUST format your entire output as a single JSONL (JSON Lines) block. Each line in the output must be a valid JSON object representing one query-configuration pair. **Do not include any text, explanations, or markdown formatting outside of the JSONL block.**

8.  **Detailed Edge Specification (Advanced):** For more precise control, you can optionally specify the exact edges using a compact format. Use the `edges` field with a space-separated list of directed edges in the format "from_idx->to_idx" (0-indexed). This field is optional - if not provided, a standard topology template will be used. **REMEMBER: All specified edges must maintain the DAG property (no cycles).**

**Example of query-configuration pairs (JSON objects) in the output:**

**Note: All edge specifications below maintain DAG property (no cycles):**

```json
{{"query": "def odd_count(lst: List[str]) -> List[str]:\n    \"\"\"Given a list of strings, where each string consists of only digits, return a list.\n    Each element i of the output should be \"the number of odd elements in the\n    string i of the input.\" where all the i's should be replaced by the number\n    of odd digits in the i'th string of the input.\n    \"\"\"\n", "agent_count": 4, "roles": ["Project Manager", "Algorithm Designer", "Programming Expert", "Test Analyst"], "topology": "Chain", "edges": "0->1 1->2 2->3"}}
{{"query": "Two trains leave San Rafael at the same time. They begin traveling westward, both traveling for 80 miles. The next day, they travel northwards, covering 150 miles. What's the distance covered by each train in the two days?", "agent_count": 3, "roles": ["Math Solver", "Mathematical Analyst", "Inspector"], "topology": "Chain", "edges": "0->1 1->2"}}
{{"query": "A car is being driven, in a straight line and at a uniform speed, towards the base of a vertical tower. The top of the tower is observed from the car and, in the process, it takes 10 minutes for the angle of elevation to change from 45° to 60°. After how much more time will this car reach the base of the tower? Choices: A)5(√3 + 1) B)6(√3 + √2) C)7(√3 – 1) D)8(√3 – 2) E)None of these", "agent_count": 4, "roles": ["Math Solver for choice question", "Mathematical Analyst for choice question", "Programming Expert for choice question", "Inspector for choice question"], "topology": "Star", "edges": "0->3 1->3 2->3"}}
{{"query": "In animal cells, which of the following represents the most likely pathway that a secretory protein takes as it is synthesized in a cell? Option A: Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER Option B: Ribosome–Golgi apparatus–rough ER–secretory vesicle–plasma membrane Option C: Plasma membrane–Golgi apparatus–ribosome–secretory vesicle–rough ER Option D: Ribosome–rough ER–Golgi apparatus–secretory vesicle–plasma membrane", "agent_count": 3, "roles": ["Knowlegable Expert", "Critic", "Doctor"], "topology": "Star", "edges": "0->2 1->2"}}
{{"query": "What were the primary economic, political, and social causes of World War I?", "agent_count": 3, "roles": ["Historian", "Knowlegable Expert", "Critic"], "topology": "Layered", "edges": "0->2 1->2"}}
{{"query": "I have been experiencing chronic back pain for the past 6 months, especially after sitting for long periods. I'm 35 years old, work at a desk job, and exercise occasionally. What treatment options would you recommend?", "agent_count": 3, "roles": ["Doctor", "Psychologist", "Critic"], "topology": "Star", "edges": "0->2 1->2"}}
{{"query": "What is the capital of Brazil?", "agent_count": 2, "roles": ["Knowlegable Expert", "Critic"], "topology": "Chain", "edges": "0->1"}}
{{"query": "Debug this Python function. It throws an index out of bounds error. Code: `def get_last_element(my_list): return my_list[len(my_list)]`", "agent_count": 3, "roles": ["Test Analyst", "Bug Fixer", "Critic"], "topology": "Mesh", "edges": "0->1 1->2 0->2"}}
{{"query": "Devise a comprehensive strategy to reduce urban traffic congestion in a major city. Consider technological solutions, public policy, and social incentives.", "agent_count": 5, "roles": ["Project Manager", "Knowlegable Expert", "Historian", "Psychologist", "Critic"], "topology": "FullConnected"}}
{{"query": "Design the architecture for a food delivery app. It needs user accounts, restaurant listings, order processing, and a real-time delivery tracking feature.", "agent_count": 6, "roles": ["Project Manager", "Algorithm Designer", "Programming Expert", "Test Analyst", "Critic", "Bug Fixer"], "topology": "Layered", "edges": "0->1 0->2 1->3 2->3 3->4 4->5"}}
```

Now, begin generating the {num_to_generate} query-configuration pairs in the specified JSONL format.
"""

def generate_dataset(output_path: str, model_name: str, num_queries: int, batch_size: int):
    """
    Generates a dataset using a GPT model by making multiple API calls in batches and saves it to a JSONL file.

    Args:
        output_path (str): The path to save the generated JSONL file.
        model_name (str): The name of the GPT model to use.
        num_queries (int): The total number of queries to generate.
        batch_size (int): The number of queries to generate in each API call.
    """
    client = OpenAI(base_url=MINE_BASE_URL, api_key=MINE_API_KEY)

    # Overwrite the file to start fresh
    with open(output_path, 'w', encoding='utf-8') as f:
        pass

    num_batches = math.ceil(num_queries / batch_size)
    print(f"Starting dataset generation for {num_queries} queries in {num_batches} batches of {batch_size}...")

    for i in range(num_batches):
        print(f"--- Generating batch {i + 1}/{num_batches} ---")

        current_batch_size = min(batch_size, num_queries - (i * batch_size))
        if current_batch_size <= 0:
            continue

        prompt = PROMPT_TEMPLATE.format(num_to_generate=current_batch_size)

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=4095,
            )

            generated_text = response.choices[0].message.content
            
            if generated_text and generated_text.strip().startswith("```json"):
                first_newline = generated_text.find('\n')
                if first_newline != -1:
                    generated_text = generated_text[first_newline+1:]
                if generated_text.strip().endswith("```"):
                    generated_text = generated_text.rsplit('```', 1)[0].strip()

            print(f"Successfully received response for batch {i + 1}. Appending to {output_path}...")

            with open(output_path, 'a', encoding='utf-8') as f:
                f.write(generated_text)
                # Ensure a newline exists between batches
                if not generated_text.endswith('\n'):
                    f.write('\n')

        except Exception as e:
            print(f"An error occurred during batch {i + 1}: {e}")
            print("Skipping this batch and continuing...")

    print(f"Dataset generation complete. Requested {num_queries} queries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a dataset using a GPT model based on a predefined prompt.")
    parser.add_argument(
        "--output_path",
        type=str,
        default="../llm_generated_data.jsonl",
        help="The path to save the generated JSONL file."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt-4o-mini",
        help="The name of the GPT model to use (e.g., 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo')."
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=500,
        help="Total number of queries to generate."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of queries to generate per API call."
    )
    args = parser.parse_args()

    generate_dataset(args.output_path, args.model_name, args.num_queries, args.batch_size)
