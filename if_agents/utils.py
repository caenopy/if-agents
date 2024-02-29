import json
import dspy

def write_to_json(data: any, filename: str):
    with open(filename, 'w') as f:
        json.dump(data, f)
        
def read_from_json(filename: str):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def write_to_file(data: any, filename: str):
    with open(filename, 'w') as f:
        f.write(data)

def write_history(filename: str, n: int = 1):
    """
    Write the last n entries of the language model history to a file.
    """
    lm = dspy.settings.lm
    file = open(filename, "a")
    provider: str = lm.provider

    last_prompt = None
    printed = []
    n = n

    for x in reversed(lm.history[-100:]):
        prompt = x["prompt"]

        if prompt != last_prompt:

            if provider == "clarifai" or provider == "google":
                printed.append(
                    (
                        prompt,
                        x['response']
                    )
                )
            else:
                printed.append(
                    (
                        prompt,
                        x["response"].generations
                        if provider == "cohere"
                        else x["response"]["choices"],
                    )
                )

        last_prompt = prompt

        if len(printed) >= n:
            break

    for idx, (prompt, choices) in enumerate(reversed(printed)):
        print("\n\n\n", file=file)
        print(prompt, end="", file=file)
        text = ""
        if provider == "cohere":
            text = choices[0].text
        elif provider == "openai" or provider == "ollama":
            text = ' ' + lm._get_choice_text(choices[0]).strip()
        elif provider == "clarifai":
            text=choices
        elif provider == "google":
            text = choices[0].parts[0].text
        else:
            text = choices[0]["text"]
        print(f"[[{text}]]", file=file)

        if len(choices) > 1:
            print(f"\t [(and {len(choices)-1} other completions)]", file=file)
        print("\n\n\n", file=file)
    
    file.close()

def get_game_list(difficulty: str):
    """
    Returns a list of games from the Jericho Game Suite, filtered by difficulty.
    `difficulty` must be one of 'possible', 'difficult', 'extreme'
    """
    METADATA_DIR = 'if_agents/jericho-metadata'
    return read_from_json(f'{METADATA_DIR}/{difficulty}.json')
    
    