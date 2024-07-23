#!/usr/bin/env python3
import multiprocessing
import os
import glob
import mimetypes
import time
import argparse
import tempfile
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from concurrent.futures import ThreadPoolExecutor, as_completed

SUPPORTED_MIME_TYPES = ['text/plain']

def get_mime_type(filepath):
    mime_type, _ = mimetypes.guess_type(filepath)
    return mime_type if mime_type in SUPPORTED_MIME_TYPES else 'text/plain'

def truncate_file(filepath, max_lines=100):
    with open(filepath, 'rb') as file:
        lines = file.readlines()[-max_lines:]
    with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.txt') as temp_file:
        temp_file.writelines(lines)
    return temp_file.name

def upload_file_with_retry(filepath, display_name, mime_type, max_retries=5, base_wait_time=5, max_lines=100, force_mime_type=None):
    relative_path = os.path.relpath(filepath)
    
    with open(filepath, 'rb') as file:
        lines = file.readlines()
        truncated_content = b''.join(lines[-max_lines:]).decode('utf-8', errors='replace')
    
    metadata = f"File: {relative_path}\nOriginal Path: {filepath}\nNote: This file has been truncated to the last {max_lines} lines.\n\n"
    full_content = metadata + truncated_content
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
        temp_file.write(full_content)
        temp_file_path = temp_file.name

    file_size = os.path.getsize(temp_file_path)
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting to upload last {max_lines} lines of {relative_path} with metadata (Size: {file_size/1024/1024:.2f}MB)")
            start_time = time.time()
            file_response = genai.upload_file(path=temp_file_path, display_name=display_name, mime_type=force_mime_type or mime_type)
            upload_time = time.time() - start_time
            print(f"Successfully uploaded truncated {relative_path} in {upload_time:.2f} seconds")
            os.unlink(temp_file_path)
            return file_response
        except Exception as e:
            print(f"Error uploading truncated {relative_path}: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                os.unlink(temp_file_path)
                raise

def count_tokens_with_retry(model, file_response, max_retries=5, base_wait_time=5):
    for attempt in range(max_retries):
        try:
            print(f"Counting tokens for {file_response.display_name}")
            token_count = model.count_tokens([file_response])
            print(f"Token count for {file_response.display_name}: {token_count.total_tokens}")
            return token_count.total_tokens
        except Exception as e:
            print(f"Error counting tokens: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = base_wait_time * (2 ** attempt)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise

def process_file(args):
    filepath, relative_path, mime_type, force_mime_type, model_name, api_key = args
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name=model_name)
        file_response = upload_file_with_retry(filepath, relative_path, mime_type, force_mime_type=force_mime_type)
        file_tokens = count_tokens_with_retry(model, file_response)
        return file_response, file_tokens, None
    except Exception as e:
        print(f"Error processing {filepath}: {str(e)}")
        return None, 0, filepath

def upload_files(directory, model, api_key, force_mime_type=None, max_workers=5):
    uploaded_files = []
    skipped_files = []
    total_tokens = 0
    max_tokens = 2000000  # 2M token limit

    files_to_process = [
        (filepath, os.path.relpath(filepath, directory), get_mime_type(filepath), force_mime_type, model.model_name, api_key)
        for filepath in glob.glob(f"{directory}/**/*", recursive=True)
        if os.path.isfile(filepath)
    ]

    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.map(process_file, files_to_process)

    for file_response, file_tokens, error_filepath in results:
        if error_filepath:
            skipped_files.append(error_filepath)
        elif total_tokens + file_tokens > max_tokens:
            print(f"Skipping {file_response.display_name}: Would exceed token limit")
            skipped_files.append(file_response.display_name)
            genai.delete_file(name=file_response.name)
        else:
            uploaded_files.append(file_response)
            total_tokens += file_tokens
            print(f"Processed file {file_response.display_name} (Tokens: {file_tokens})")
            print(f"Total tokens used: {total_tokens}/{max_tokens}")

    return uploaded_files, skipped_files, total_tokens

def cleanup_files(uploaded_files, max_workers=10):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(genai.delete_file, name=file.name) for file in uploaded_files]
        for future, file in zip(futures, uploaded_files):
            try:
                future.result()
                print(f'Deleted file {file.display_name}')
            except Exception as e:
                print(f"Failed to delete {file.display_name}: {str(e)}")

def chat_with_model(model, system_prompt, uploaded_files):
    chat = model.start_chat(history=[])
    print("Chat started. Type 'exit' to end the conversation.")
    print("AI: Hello! I'm ready to help you with any questions about the uploaded files. What would you like to know?")

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                break
            
            full_prompt = [
                system_prompt,
                f"User question: {user_input}",
                "Please analyze the attached files to answer the question. Each file begins with metadata including its file path.",
                "The following files are available for reference:",
                *[f"- {file.display_name}" for file in uploaded_files],
            ] + uploaded_files

            print("AI: ", end="", flush=True)
            response = chat.send_message(full_prompt, stream=True)
            for chunk in response:
                print(chunk.text, end="", flush=True)
            print()  # New line after the complete response

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Continuing chat...")

def list_uploaded_files():
    try:
        return genai.list_files()
    except Exception as e:
        print(f"An error occurred while listing files: {e}")
        return []

def delete_file(file_id):
    try:
        genai.delete_file(file_id)
        print(f"File with ID {file_id} has been successfully deleted.")
    except Exception as e:
        print(f"An error occurred while deleting file {file_id}: {e}")

def cleanup_all_files(max_workers=10):
    files = list_uploaded_files()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(delete_file, file.name) for file in files]
        for future in as_completed(futures):
            future.result()  # This will raise an exception if the deletion failed

def main():
    parser = argparse.ArgumentParser(description="Upload files and chat with Gemini model.")
    parser.add_argument("directory", help="Directory containing files to upload")
    parser.add_argument("--force-mime", help="Force all files to use this MIME type")
    parser.add_argument("--max-workers", type=int, default=25, help="Maximum number of parallel uploads")
    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")

    genai.configure(api_key=api_key)
    model_name = "models/gemini-1.5-pro-latest"
    model = genai.GenerativeModel(model_name=model_name)

    print("Cleaning up any existing files...")
    cleanup_all_files(max_workers=args.max_workers)

    uploaded_files, skipped_files, total_tokens = upload_files(args.directory, model, api_key, force_mime_type=args.force_mime, max_workers=args.max_workers)

    if total_tokens >= 2000000:
        print("Warning: Maximum token limit reached. Some files may have been skipped.")

    if skipped_files:
        print("The following files were skipped:")
        for file in skipped_files:
            print(f"- {file}")

    system_prompt = ("You are an AI assistant with access to several uploaded files. "
                     "When the user asks a question, you can analyze and refer to these files to provide information. "
                     "The files may contain logs, configuration data, or other relevant information.")

    chat_with_model(model, system_prompt, uploaded_files)

    print("Cleaning up uploaded files...")
    cleanup_files(uploaded_files, max_workers=args.max_workers)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Needed for Windows compatibility
    main()
