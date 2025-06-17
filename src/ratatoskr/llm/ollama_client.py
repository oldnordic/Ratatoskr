"""
Ollama LLM client integration and management.

This module provides a client interface for interacting with Ollama-served
language models. Currently implemented as a placeholder, it's designed to
be extended with full Ollama API integration for model management and
inference.

Key Features:
- Ollama client interface
- Model management capabilities
- Connection handling
- Error management
- Future: Full Ollama API integration

Future Enhancements:
- Complete Ollama API integration
- Model listing and management
- Streaming response support
- Model switching capabilities
- Performance monitoring
"""

import logging
import requests
from typing import Optional, Dict, Any, List

# Configuration constants
DEFAULT_TIMEOUT = 30  # seconds
DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaClient:
    """
    Client for interacting with Ollama API.
    
    Provides methods for connecting to Ollama, listing models,
    loading models, and generating text responses.
    """
    
    def __init__(self, base_url: str = DEFAULT_BASE_URL) -> None:
        """
        Initialize the Ollama client.
        
        Args:
            base_url: Base URL for Ollama API
        """
        self.base_url = base_url.rstrip('/')
        self.is_connected = False
        self.model_name = None
        self.available_models = []
        self.session = requests.Session()
        
        logging.info(f"OllamaClient initialized with base URL: {self.base_url}")
    
    def connect(self) -> bool:
        """
        Connect to Ollama instance and verify availability.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            logging.info("Connecting to Ollama instance...")
            
            # Test connection by fetching available models
            response = self.session.get(f"{self.base_url}/api/tags", timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            # Parse available models
            models_data = response.json()
            self.available_models = [model["name"] for model in models_data.get("models", [])]
            
            self.is_connected = True
            logging.info(f"Successfully connected to Ollama. Available models: {self.available_models}")
            return True
            
        except requests.RequestException as e:
            logging.error(f"Failed to connect to Ollama: {e}")
            self.is_connected = False
            return False
        except Exception as e:
            logging.error(f"Unexpected error connecting to Ollama: {e}")
            self.is_connected = False
            return False
    
    def disconnect(self) -> None:
        """Disconnect from Ollama instance."""
        try:
            logging.info("Disconnecting from Ollama...")
            self.session.close()
            self.is_connected = False
            self.model_name = None
            self.available_models = []
            logging.info("Disconnected from Ollama")
        except Exception as e:
            logging.error(f"Error disconnecting from Ollama: {e}")
    
    def list_models(self) -> List[str]:
        """
        Get list of available models.
        
        Returns:
            List[str]: List of available model names
        """
        try:
            if not self.is_connected:
                logging.warning("Not connected to Ollama")
                return []
            
            response = self.session.get(f"{self.base_url}/api/tags", timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            models_data = response.json()
            self.available_models = [model["name"] for model in models_data.get("models", [])]
            
            logging.info(f"Retrieved {len(self.available_models)} available models")
            return self.available_models.copy()
            
        except requests.RequestException as e:
            logging.error(f"Error listing models: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error listing models: {e}")
            return []
    
    def load_model(self, model_name: str) -> bool:
        """
        Load a specific model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if not self.is_connected:
                logging.warning("Not connected to Ollama")
                return False
            
            logging.info(f"Loading model: {model_name}")
            
            # Send pull request to ensure model is available
            pull_data = {"name": model_name}
            response = self.session.post(f"{self.base_url}/api/pull", json=pull_data, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            self.model_name = model_name
            logging.info(f"Successfully loaded model: {model_name}")
            return True
            
        except requests.RequestException as e:
            logging.error(f"Error loading model {model_name}: {e}")
            return False
        except Exception as e:
            logging.error(f"Unexpected error loading model {model_name}: {e}")
            return False
    
    def generate(self, prompt: str, **kwargs: Any) -> Optional[str]:
        """
        Generate text using the loaded model.
        
        Args:
            prompt: Input prompt for generation
            **kwargs: Additional generation parameters
            
        Returns:
            Optional[str]: Generated text, or None if generation failed
        """
        try:
            if not self.is_connected:
                logging.warning("Not connected to Ollama")
                return None
            
            if not self.model_name:
                logging.warning("No model loaded")
                return None
            
            logging.info(f"Generating text with model: {self.model_name}")
            
            # Prepare generation request
            generation_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
            
            # Add optional parameters
            if "temperature" in kwargs:
                generation_data = {
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": kwargs["temperature"]}
                }
            
            response = self.session.post(f"{self.base_url}/api/generate", json=generation_data, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            result = response.json()
            generated_text = result.get("response", "")
            
            logging.info("Text generation completed")
            return generated_text
            
        except requests.RequestException as e:
            logging.error(f"Error generating text: {e}")
            return None
        except Exception as e:
            logging.error(f"Unexpected error generating text: {e}")
            return None
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Name of the model (uses current model if None)
            
        Returns:
            Dict[str, Any]: Model information
        """
        try:
            target_model = model_name or self.model_name
            
            if not target_model:
                return {"error": "No model specified"}
            
            # Get model information
            response = self.session.post(f"{self.base_url}/api/show", json={"name": target_model}, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            
            model_info = response.json()
            
            return {
                "name": target_model,
                "loaded": target_model == self.model_name,
                "parameters": model_info.get("parameter_size", "Unknown"),
                "size": model_info.get("size", "Unknown"),
                "modified_at": model_info.get("modified_at", "Unknown"),
                "license": model_info.get("license", "Unknown")
            }
            
        except requests.RequestException as e:
            logging.error(f"Error getting model info: {e}")
            return {"error": str(e)}
        except Exception as e:
            logging.error(f"Unexpected error getting model info: {e}")
            return {"error": str(e)}
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get information about the client status.
        
        Returns:
            dict: Client status and configuration information
        """
        return {
            "base_url": self.base_url,
            "is_connected": self.is_connected,
            "current_model": self.model_name,
            "available_models": len(self.available_models),
            "implementation": "full API integration"
        }


def get_llm_client() -> Optional[OllamaClient]:
    """
    Get or create an Ollama client instance.
    
    Returns:
        Optional[OllamaClient]: Client instance, or None if creation failed
    """
    try:
        client = OllamaClient()
        if client.connect():
            return client
        else:
            logging.warning("Failed to connect to Ollama")
            return None
    except Exception as e:
        logging.error(f"Error creating Ollama client: {e}")
        return None


def test_ollama_connection() -> Dict[str, Any]:
    """
    Test connection to Ollama and return status information.
    
    Returns:
        dict: Connection test results
    """
    try:
        client = OllamaClient()
        
        # Test connection
        connected = client.connect()
        
        if connected:
            # Test model listing
            models = client.list_models()
            
            result = {
                "connected": True,
                "available_models": models,
                "base_url": client.base_url
            }
        else:
            result = {
                "connected": False,
                "error": "Failed to connect to Ollama",
                "base_url": client.base_url
            }
        
        client.disconnect()
        return result
        
    except Exception as e:
        logging.error(f"Error testing Ollama connection: {e}")
        return {
            "connected": False,
            "error": str(e)
        }
