import ollama
import config
import logging

def get_ai_response(history: list):
    """
    Sends the conversation history to the Ollama model and gets a response.
    
    Args:
        history (list): A list of message dictionaries.

    Returns:
        str: The AI's response content.
    """
    try:
        logging.info(f"Sending request to Ollama model: {config.MODEL_NAME}")
        response = ollama.chat(
            model=config.MODEL_NAME,
            messages=history
        )
        logging.info("Received response from Ollama successfully.")
        return response['message']['content']
    except Exception as e:
        logging.error(f"Error communicating with Ollama: {e}")
        # Propagate the error so the main thread can catch it
        raise e