#!/usr/bin/env python3
import os
import random
import string
import argparse
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
import time

def generate_random_content(size=1000):
    """Generate random text content of approximately 'size' characters."""
    return ''.join(random.choices(string.ascii_letters + string.digits + ' \n', k=size))

def create_test_files(directory, num_files, file_size=1000):
    """Create a specified number of small text files with random content."""
    os.makedirs(directory, exist_ok=True)
    for i in range(num_files):
        filename = f"test_file_{i}.txt"
        filepath = os.path.join(directory, filename)
        with open(filepath, 'w') as f:
            f.write(generate_random_content(file_size))
    print(f"Created {num_files} files in {directory}")

def upload_files(directory, model):
    uploaded_files = []
    skipped_files = []
    total_tokens = 0
    max_tokens = 2000000  # 2M token limit

    start_time = time.time()
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                file_response = genai.upload_file(path=filepath, display_name=filename)
                token_count = model.count_tokens([file_response])
                file_tokens = token_count.total_tokens

                if total_tokens + file_tokens > max_tokens:
                    print(f"Skipping {filename}: Would exceed token limit")
                    skipped_files.append(filename)
                    genai.delete_file(name=file_response.name)
                    continue

                uploaded_files.append(file_response)
                total_tokens += file_tokens
                print(f"Uploaded {filename} (Tokens: {file_tokens})")
            except Exception as e:
                print(f"Failed to upload {filename}: {str(e)}")
                skipped_files.append(filename)

    end_time = time.time()
    print(f"\nUpload Summary:")
    print(f"Total files uploaded: {len(uploaded_files)}")
    print(f"Files skipped: {len(skipped_files)}")
    print(f"Total tokens: {total_tokens}")
    print(f"Total upload time: {end_time - start_time:.2f} seconds")

    return uploaded_files, skipped_files, total_tokens

def cleanup_files(uploaded_files):
    for file in uploaded_files:
        try:
            genai.delete_file(name=file.name)
            print(f'Deleted file {file.display_name}')
        except Exception as e:
            print(f"Failed to delete {file.display_name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Generate and upload multiple small files to test Gemini API.")
    parser.add_argument("--num_files", type=int, default=100, help="Number of files to generate")
    parser.add_argument("--file_size", type=int, default=1000, help="Approximate size of each file in characters")
    parser.add_argument("--directory", default="test_files", help="Directory to store generated files")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")

    create_test_files(args.directory, args.num_files, args.file_size)
    uploaded_files, skipped_files, total_tokens = upload_files(args.directory, model)

    # Attempt a simple query
    if uploaded_files:
        system_prompt = "You are an AI assistant analyzing multiple small text files."
        chat = model.start_chat(history=[])
        try:
            response = chat.send_message([
                system_prompt,
                "How many files were successfully uploaded?",
                "Please provide a brief summary of the upload process."
            ] + uploaded_files)
            print("\nAI Response:")
            print(response.text)
        except Exception as e:
            print(f"Error querying the model: {str(e)}")

    cleanup_files(uploaded_files)

if __name__ == "__main__":
    main()
