import pandas as pd
import requests
from rich.progress import Progress


def query_ollama(prompt):
    """
    Send a query to Ollama LLM and return the response.
    """
    response = requests.post(
        "http://localhost:11434/api/generate",  # Update this if the Ollama API URL differs
        json={"model": "llama3.2", "prompt": prompt, "stream": False},
    )
    if response.status_code == 200:
        return response.json()["response"].strip().lower()
    else:
        raise Exception(f"Error from Ollama: {response.status_code}, {response.text}")


def main():
    # Load the CSV files
    descriptions = pd.read_csv("descriptions.csv")
    wiki_tracker = pd.read_csv("rossmann_wiki_tracker.csv")

    # Merge descriptions into wiki_tracker
    merged = wiki_tracker.merge(
        descriptions[["video_title", "description"]], on="video_title", how="left"
    )

    # Define a function to process each row
    def process_row(row):
        prompt = (
            f"Your task is to determine whether a given video title and description indicate content "
            f"that requires a wiki article. A wiki article should be created if the content covers a "
            f"specific event, policy, invention, or newsworthy subject that provides value in documentation "
            f"and wider dissemination. Examples of when to answer 'yes' or 'no' are provided below.\n\n"
            f"Examples:\n"
            f"1. 'Cat video' -> No\n"
            f"2. 'Random rant' -> No\n"
            f"3. 'Discussion of specific politician screwing right to repair bill after receiving $3300 from AT&T lobbyist' -> Yes\n"
            f"4. 'Ford filing patent on how to use in-car spyware to advertise to passengers during drive' -> Yes\n\n"
            f"Now, analyze the following information:\n\n"
            f"Video Title: {row['video_title']}\n"
            f"Description: {row['description']}\n\n"
            f"Does this content require a wiki article? Please answer only with 'yes' or 'no'."
        )
        try:
            result = query_ollama(prompt)
            return result
        except Exception as e:
            print(f"Error processing row: {e}")
            return "error"

    # Use Rich Progress to track progress through the rows
    with Progress() as progress:
        task = progress.add_task("[cyan]Processing rows...", total=len(merged))
        results = []

        for _, row in merged.iterrows():
            result = process_row(row)
            results.append(result)
            progress.update(task, advance=1)

    # Add results to the DataFrame
    merged["needs_wiki_article"] = results

    # Write the updated DataFrame to a CSV file
    merged.to_csv("merged_first_try.csv", index=False)
    print("Processing complete. Results saved to 'merged_first_try.csv'.")


if __name__ == "__main__":
    main()
