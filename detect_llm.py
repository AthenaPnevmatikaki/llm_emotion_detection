import requests
import json
import pandas as pd
import time


# Function to analyze the emotions in a single sentence
def analyze_emotion(model, content, max_tokens, sentence):
    url = "http://localhost:1234/v1/chat/completions"
    headers = {
        "Content-Type": "application/json"
    }
    data = {"model": model, "messages": []}
    data["messages"].append({"role": "system", "content": content})
    data["messages"].append({"role": "user", "content": sentence})
    data["max_tokens"] = max_tokens
    data["temperature"] = 0.2
    # Send the request
    response = requests.post(url, headers=headers, data=json.dumps(data))
    # Convert response to JSON
    response = response.json()
    # Return the content of the response
    response_text = response['choices'][0]['message']['content']
    # Extract content between the first square brackets
    start_idx = response_text.find('[')
    end_idx = response_text.find(']', start_idx) + 1  # +1 to include the closing bracket
    if start_idx != -1 and end_idx != -1:
        # Return the content inside the square brackets
        return response_text[start_idx:end_idx]
    else:
        # Return the full response if the brackets are not found
        return response_text


# Function to read sentences from a CSV file and analyze each
def analyze_emotions_from_csv(file_path, content=1, model_id=0, output="JSON", start_index=0):
    if content == 0:
        max_tokens = 1000
        result_type = 'reason'
        content_text = """Return the emotion in the given sentences.
            Select between empty, sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.
            Reply only with a list of the emotions you are most confident that are in the text, their strength and the reasons you selected them.
            Structure the returned list as JSON using: [{"emotion": "detected emotion", "strength": "medium/high", "rationale": "your explanation"]}, ...]"""
    elif content == 1:
        max_tokens = 1000
        result_type = 'reason'
        content_text = """Return the most evident emotion in the given sentences.
            Reply only with one emotion from this list: sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, and anger.
            Include your reason for selecting this emotion.
            Structure the returned list as JSON using: [{"emotion": "detected emotion", "rationale": "your explanation"]}, ...]"""
    elif content == 2:
        max_tokens = 5
        result_type = 'txt'
        content_text = """Detect the most prominent emotion in the sentence I will give you, by selecting the emotion from the following list:
            sadness, enthusiasm, neutral, worry, surprise, love, fun, hate, happiness, boredom, relief, anger
            Your reply must be just one word selected from the above list. Do not use anything else!"""
    else:
        max_tokens = 1
        result_type = 'num'
        content_text = """Detect the most prominent emotion expressed in the text I will give you.
            Use only a number from the list 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, and 12 in your response to represent the detected emotion, where the returned numbers
            correspond to detected emotions using the following mapping:
                1. sadness
                2. enthusiasm
                3. neutral
                4. worry
                5. surprise
                6. love
                7. fun
                8. hate
                9.happiness
                10. boredom
                11. relief
                12. anger
            Do not use any words in your response, just the number corresponding to the most prominent emotion"""
    model = "llama-3.2-1b@q8"
    if model_id == 1:
        model = "llama-3.2-1b@q8"
    elif model_id == 2:
        model = "emollama-chat-7b"
    elif model_id == 3:
        model = "llama-3.2-3b@q4"
    elif model_id == 4:
        model = "llama-3.2-3b@q8"
    elif model_id == 5:
        model = "llama-3.2-8b@q4"
    if output == "CSV":
        extension = "csv"
    else:
        extension = "txt"
    df = pd.read_csv(file_path)
    number_of_sentences = len(df.index)
    total_time = 0
    # Process each sentence in the 'content' column
    for index, row in df.iterrows():
        if index < start_index:
            continue
        sentence = row['content'].strip()  # Remove any extra whitespace
        if sentence:  # Ensure the sentence is not empty
            start_time = time.time()  # Record start time
            result = analyze_emotion(model, content_text, max_tokens, sentence)
            if output == "JSON":
                result = "{" + f'"sentence": "{sentence}",\n"emotions": {result}' + "}"
            else:
                result = f'"{sentence}","{result.lower()}"\n'
            end_time = time.time()  # Record end time
            elapsed_time = end_time - start_time  # Calculate the time taken for this iteration
            total_time += elapsed_time
            remaining_sentences = number_of_sentences - index - 1
            time_per_sentence = total_time / (index + 1 - start_index)
            remaining_time = time_per_sentence * remaining_sentences / 3600
            if index % 10 == 0:
                print(f'{index + 1}/{number_of_sentences} sentences: Time per sentence {time_per_sentence:.3f}"'
                      f", {remaining_time:.3f} hours remaining")
            with open(f"results_{result_type}_{model}.{extension}", 'a', encoding="charmap", errors="replace") as file:
                if output == "JSON" and file.tell() == 0:  # Checks if the file is empty
                    file.write("[\n")
                # Add commas and format JSON entries properly
                if output == "JSON" and index > 0:
                    file.write(",\n")  # Separate entries with a comma
                # Start writing JSON array if the file is empty
                file.write(result)
    # Close the JSON array properly after processing all sentences
    if output == "JSON":
        with open(f"results_{result_type}_{model}.{extension}", 'a', encoding="charmap", errors="replace") as file:
            file.write("\n]\n")


# Usage
# analyze_emotions_from_csv("tweet_emotions.csv", content=1, model_id=0, output="JSON", start_index=12050)
analyze_emotions_from_csv("tweet_emotions.csv", content=2, model_id=4, output="CSV", start_index=0)
