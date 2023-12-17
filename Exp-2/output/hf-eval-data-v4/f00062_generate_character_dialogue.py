# requirements_file --------------------

!pip install -U transformers torch

# function_import --------------------

from transformers import AutoTokenizer, AutoModelForCausalLM

# function_code --------------------


def generate_character_dialogue(character_persona, dialogue_history, user_input):
    """
    Generate dialogue as a fictional character based on user input.

    Parameters:
        character_persona (str): Description of the character's persona.
        dialogue_history (str): History of the conversation.
        user_input (str): The user's input message.

    Returns:
        str: The fictional character's generated dialogue.
    """
    tokenizer = AutoTokenizer.from_pretrained('waifu-workshop/pygmalion-6b')
    model = AutoModelForCausalLM.from_pretrained('waifu-workshop/pygmalion-6b')

    input_text = f"{character_persona}\n<START>\n{dialogue_history}\nYou: {user_input}\nCharacter:"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return output_text


# test_function_code --------------------


def test_generate_character_dialogue():
    character_persona = "A smart and witty detective with a sharp sense of humor."
    dialogue_history = "Detective: There's been a murder at the mansion.\nYou: Who do you think did it?"
    user_input = "Could it be the butler?"

    result = generate_character_dialogue(character_persona, dialogue_history, user_input)
    assert isinstance(result, str), "Output should be a string."
    assert result.startswith('Detective:'), "Response should start with 'Detective:'"

    print("Test passed for generate_character_dialogue function.")

test_generate_character_dialogue()
