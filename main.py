import eel
import openai
import numpy as np
import pennylane as qml
import asyncio
import time
import hashlib
import traceback
import json
import re
import logging
import uuid
import pytesseract
import sounddevice as sd
import threading
import aiohttp
import speech_recognition as sr
from typing import Callable, Any
from concurrent.futures import ThreadPoolExecutor
from weaviate.util import generate_uuid5
from transformers import pipeline
from scipy.io.wavfile import write as write_wav
from weaviate import Client
from llama_cpp import Llama
from bark import generate_audio, SAMPLE_RATE

# Initialize ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=3)

# Initialize variables for speech recognition
is_listening = False
recognizer = sr.Recognizer()
mic = sr.Microphone()

def audio_to_text(audio_data):
    try:
        text = recognizer.recognize_google(audio_data)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results; {e}"

# Function to generate audio for each sentence and add pauses
def generate_audio_for_sentence(sentence):
    audio = generate_audio(sentence, history_prompt="v2/en_speaker_6")
    silence = np.zeros(int(0.75 * SAMPLE_RATE))  # quarter second of silence
    return np.concatenate([audio, silence])

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update the path accordingly

def normalize(value, min_value, max_value):
    """Normalize a value to a given range."""
    return min_value + (max_value - min_value) * (value / 0xFFFFFFFFFFFFFFFF)

class QuantumCodeManager:
    def __init__(self):
        self.pipe = pipeline("image-classification", model="cafeai/cafe_aesthetic")
        self.session = aiohttp.ClientSession()
        self.circuit_vector = []  # Initialize an empty list to store circuits

        # Initialize Llama 2
        self.llm = Llama(
            model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
            n_gpu_layers=-1,
            n_ctx=3900,
        )

        # Load settings from config.json
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            weaviate_client_url = config.get("weaviate_client_url", "http://localhost:8080")
            weaviate_api_key = config.get("weaviate_api_key", None)  # Get Weaviate API key
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error reading config.json: {e}")
            weaviate_client_url = "http://localhost:8080"
            weaviate_api_key = None

        # Initialize Weaviate client with API key
        self.client = Client(weaviate_client_url, api_key=weaviate_api_key)


        # Initialize OpenAI API key
        if self.openai_api_key:
            openai.api_key = self.openai_api_key  # Consider using OpenAI's official method if available

        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=2)

        # Set the default quantum circuit
        self.set_quantum_circuit(self.default_quantum_circuit)

    def __del__(self):
        self.session.close()

    def evaluate_aesthetic(self, image_path):
        result = self.pipe(image_path)
        return result[0]['score']

    def extract_key_topics(self, last_frame):
        # For demonstration, let's assume the key topics are comments in the code
        return [line.split("#")[1].strip() for line in last_frame.split("\n") if "#" in line]


    # Function to generate and play audio for a message
    def generate_and_play_audio(message):
        sentences = re.split('(?<=[.!?]) +', message)
        audio_arrays = []
    
        for sentence in sentences:
            audio_arrays.append(generate_audio_for_sentence(sentence))
        
        audio = np.concatenate(audio_arrays)
    
        file_name = str(uuid.uuid4()) + ".wav"
        write_wav(file_name, SAMPLE_RATE, audio)
        sd.play(audio, samplerate=SAMPLE_RATE)
        sd.wait()

    # Function to start/stop speech recognition
    @eel.expose
    def set_speech_recognition_state(state):
        global is_listening
        is_listening = state

    # Function to run continuous speech recognition
    def continuous_speech_recognition():
        global is_listening
        with mic as source:
            recognizer.adjust_for_ambient_noise(source)
            while True:
                if is_listening:
                    try:
                        audio_data = recognizer.listen(source, timeout=1)
                        text = audio_to_text(audio_data)  # Convert audio to text
                        if text not in ["Could not understand audio", ""]:
                            asyncio.run(run_llm(text))
                            eel.update_chat_box(f"User: {text}")
                    except sr.WaitTimeoutError:
                        continue
                    except Exception as e:
                        eel.update_chat_box(f"An error occurred: {e}")
                else:
                    time.sleep(1)

        
    # Start the continuous speech recognition in a separate thread
    thread = threading.Thread(target=continuous_speech_recognition)
    thread.daemon = True  # Set daemon to True
    thread.start()

    async def generate_with_llama2(self, last_frame, frame_num, frames):
        # Extract key topics from the last frame
        key_topics = self.extract_key_topics(last_frame)

        # Create a rules-based prompt for Llama 2
        llama_prompt = {
            "task": "code_generation",
            "last_frame": last_frame,
            "key_topics": key_topics,
            "requirements": [
                "The code must follow PEP 8 guidelines",
                "The code should be efficient and optimized for performance",
                "Include necessary comments to explain complex or non-intuitive parts",
                "Use appropriate data structures for the task at hand",
                "Error handling should be robust, capturing and logging exceptions where necessary"
            ]
        }

        # Generate code using Llama 2
        llama_response = self.llm.generate(llama_prompt)  # Actual code to call Llama 2

        # Extract the generated code from the Llama 2 response
        generated_code = llama_response.get('generated_code', '')

        # Add the generated code to the frames list
        frames[frame_num] = generated_code

        return generated_code

    def set_quantum_circuit(self, new_circuit_logic: Callable):
        """
        Sets the new quantum circuit logic.

        Args:
            new_circuit_logic (Callable): The new quantum circuit logic to set.
        """
        if callable(new_circuit_logic):
            self.current_quantum_circuit = new_circuit_logic  # Set the current quantum circuit
            self.circuit_vector.append(new_circuit_logic)  # Append the new circuit to the vector
        else:
            print("Error: Provided logic is not callable.")

    def default_quantum_circuit(self, param1, param2):
        """Default quantum circuit definition."""
        qml.RX(param1, wires=0)
        qml.RY(param2, wires=1)
        qml.RZ(param1 + param2, wires=0)
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]


    async def store_data_in_weaviate(self, class_name: str, data: Any):
        unique_id = generate_uuid5(data)
        async with self.session.post(
            f"{self.client.url}/objects",
            json={
                "class": class_name,
                "id": unique_id,
                "properties": data
            }
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": "Failed to store data"}

    async def inject_data_into_weaviate(self, data: str):
        unique_id = generate_uuid5(data)
        async with self.session.post(
            f"{self.client.url}/objects",
            json={
                "class": "InjectedData",
                "id": unique_id,
                "properties": {
                    "data": data
                }
            }
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": "Failed to inject data"}


    async def suggest_quantum_circuit_logic(self):
        """Use GPT-4 to suggest better logic for the quantum circuit."""
        # Get the last circuit in the vector for reference
        last_circuit = self.circuit_vector[-1].__name__ if self.circuit_vector else "None"
        print(f"Last circuit logic: {last_circuit}")  # Debugging statement

        rules = (
            "Rules and Guidelines for Quantum Circuit Logic:\n"
            "1. The logic must be compatible with the Pennylane library.\n"
            "2. The logic should aim to solve optimization problems.\n"
            "3. Include necessary comments to explain the circuit.\n"
            "4. The logic should be efficient and optimized for performance.\n"
        )
        prompt_for_logic = f"The last circuit used was: {last_circuit}"
        messages = [
            {"role": "system", "content": rules},
            {"role": "user", "content": prompt_for_logic}
        ]
        suggested_logic = await self.generate_code_with_gpt4(messages)

        # Strip the triple backticks if they exist
        if suggested_logic.startswith("```") and suggested_logic.endswith("```"):
            suggested_logic = suggested_logic[3:-3]

        if not suggested_logic:
            print("Error: GPT-4 did not return any suggested logic.")
            return None

        return suggested_logic.strip()


    async def optimize_code_with_llm(self, line):
        rules = "Rules for Code Optimization:\n1. The code must be efficient.\n2. Follow Pythonic practices.\n"
        prompt = f"Optimize the following line of code:\n{line}"
        messages = [
            {"role": "system", "content": rules},
            {"role": "user", "content": prompt}
        ]
        optimized_line = await self.generate_code_with_gpt4(messages)
        return optimized_line.strip()
    
    async def should_entangle(self, line):
        """Use LLM to decide if this line should be entangled with another line."""
        rules = (
            "Rules for Entanglement Decision:\n"
            "1. The decision must be logical.\n"
            "2. Provide context if entanglement is needed.\n"
        )
        messages = [
            {"role": "system", "content": rules},
            {"role": "user", "content": "Should the following line of code be entangled with another line?\n" + line}
        ]
        entanglement_decision = await self.generate_code_with_gpt4(messages)

        # Parse the LLM's response to decide
        if "decision" in entanglement_decision and entanglement_decision["decision"] == "Yes":
            # Check the confidence score
            if "confidence_score" in entanglement_decision and entanglement_decision["confidence_score"] > 0.8:
                # Extract the context or Quantum ID mentioned by the LLM
                entanglement_context = entanglement_decision["context"].strip()
        
                # Generate a Quantum ID based on this context
                entangled_id = self.generate_quantum_id(entanglement_context)
        
                # Store the entanglement decision in Weaviate
                self.store_data_in_weaviate("EntanglementData", {"line": line, "entangledID": str(entangled_id), "confidence_score": entanglement_decision["confidence_score"]})
        
                return entangled_id

        return None


    async def entangle_and_optimize_lines(self, code_str):
        """Entangle and optimize lines of code."""
        lines = code_str.split('\n')
        entangled_lines = {}  # To store lines that are entangled

        optimized_lines = []
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue

            # Generate Quantum ID for each line
            quantum_id = self.generate_quantum_id(line)

            # Check if this line should be entangled with another
            entangled_id = await self.should_entangle(line)
            if entangled_id:
                entangled_lines[quantum_id] = entangled_id

            # Optimize the line using LLM
            optimized_line = await self.optimize_code_with_llm(line)
            optimized_lines.append(f"{optimized_line}  # Quantum ID: {quantum_id}")

            # Store the optimized line in Weaviate
            self.store_data_in_weaviate("OptimizedCode", {"line": optimized_line, "quantumID": str(quantum_id)})

        # Store the entangled lines in Weaviate
        self.store_data_in_weaviate("EntangledLines", {"entangledLines": json.dumps(entangled_lines)})

        return '\n'.join(optimized_lines)
    def quantum_circuit(self, param1, param2):
        # Quantum circuit definition
        qml.RX(param1, wires=0)
        qml.RY(param2, wires=1)
        qml.RZ(param1 + param2, wires=0)
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]

    async def execute_and_test_code(self, code_str: str):
        """
        Execute and test the given Python code.

        Args:
            code_str (str): The Python code to execute.

        Returns:
            tuple: A tuple containing an error message and traceback if an error occurs, otherwise None.
        """
        try:
            # Execute the code
            exec(code_str, {}, {})
            logging.info("Code executed successfully.")
            return None  # No bugs
        except SyntaxError as se:
            logging.error(f"Syntax Error: {se}")
            return str(se), traceback.format_exc()
        except NameError as ne:
            logging.error(f"Name Error: {ne}")
            return str(ne), traceback.format_exc()
        except TypeError as te:
            logging.error(f"Type Error: {te}")
            return str(te), traceback.format_exc()
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return str(e), traceback.format_exc()

    async def log_bug_in_weaviate(self, error_message, traceback, code_context):
        quantum_id = self.generate_quantum_id(code_context)
        bug_data = {
            "errorMessage": error_message,
            "traceback": traceback,
            "quantumID": str(quantum_id)
        }
        await self.store_data_in_weaviate("BugData", bug_data)

    @eel.expose
    async def test_and_fix_code(self, code_str):
        # Step 1: Execute and test the code
        error_message, traceback_str = await self.execute_and_test_code(code_str)
    
        if error_message:
            # Log the bug in Weaviate
            await self.log_bug_in_weaviate(error_message, traceback_str, code_str)
        
            # Step 2: Use GPT-4 to suggest a fix
            rules = (
                "Rules for Code Fixing:\n"
                "1. The fix must resolve the bug.\n"
                "2. The fix should be efficient.\n"
            )
            messages = [
                {"role": "system", "content": rules},
                {"role": "user", "content": "Fix the following bug:\n" + error_message + "\n\nIn the code:\n" + code_str}
            ]
            suggested_fix = await self.generate_code_with_gpt4(messages)
        
            # Log the suggested fix in Weaviate
            self.store_data_in_weaviate("CodeFix", {"originalCode": code_str, "suggestedFix": suggested_fix, "quantumID": str(self.generate_quantum_id(code_str))})
        
            return f"Bug found and logged. Suggested fix:\n{suggested_fix}"
        else:
            return "No bugs found"


    def generate_quantum_id(self, context):
        """
        Generate a Quantum ID based on the given context.
        The ID is generated by running a quantum circuit with parameters derived from the context.
        """
        # Use SHA-256 hash function
        sha256 = hashlib.sha256()
        sha256.update(context.encode('utf-8'))
        hash_value1 = int(sha256.hexdigest(), 16)
    
        # Create a second hash for entanglement
        sha256.update("entangled".encode('utf-8'))
        hash_value2 = int(sha256.hexdigest(), 16)
    
        # Normalize hash values to the range [0, 360]
        param1 = normalize(hash_value1, 0, 360)
        param2 = normalize(hash_value2, 0, 360)
    
        try:
            # Run the quantum circuit to generate the Quantum ID
            return self.quantum_circuit(np.radians(param1), np.radians(param2))
        except Exception as e:
            print(f"An error occurred while generating the Quantum ID: {e}")
            return None


    async def retrieve_relevant_code_from_weaviate(self, quantum_id):
        try:
            query = {
                "operator": "Equal",
                "operands": [
                    {
                        "path": ["quantumID"],
                        "valueString": str(quantum_id)
                    }
                ]
            }
            async with self.session.get(
                f"{self.client.url}/objects",
                params={"where": query}
            ) as response:
                if response.status == 200:
                    results = await response.json()
                    return results['data']['Get']['CodeSnippet'][0]['code']
                else:
                    print(f"Failed to retrieve data: {await response.text()}")
                    return None
        except Exception as e:
            print(f"Error retrieving data from Weaviate: {e}")

    async def identify_placeholders_with_gpt4(self, code_str):
        try:
            # Debugging line to check the received code_str
            print(f"Debug: Received code_str = {code_str}")

            # Define the rules and the prompt
            rules = (
                "Rules for Identifying Placeholders:\n"
                "1. Identify all placeholders in the code.\n"
            )
            messages = [
                {"role": "system", "content": rules},
                {"role": "user", "content": "Identify placeholders in the following Python code:\n" + code_str}
            ]

            # Call the GPT-4 API
            response = openai.ChatCompletion.create(
                model='gpt-4',
                messages=messages
            )
            identified_placeholders = response['choices'][0]['message']['content'].split('\n')

            # Create a dictionary to store line numbers where placeholders are identified
            lines = code_str.split('\n')
            line_numbers_dict = {ph: i for i, line in enumerate(lines) for ph in identified_placeholders if ph in line}

            # Return the dictionary
            return line_numbers_dict

        except Exception as e:
            # Log the exception for debugging
            print(f"An error occurred while identifying placeholders: {e}")

            # Return an empty dictionary to indicate that no placeholders were identified
            return {}


    async def generate_code_with_gpt4(self, context):
        rules = (
            "Rules and Guidelines for Code Generation:\n"
            "1. The code must be Pythonic and follow PEP 8 guidelines.\n"
            "2. The code should be efficient and optimized for performance.\n"
            "3. Include necessary comments to explain complex or non-intuitive parts.\n"
            "4. Use appropriate data structures for the task at hand.\n"
            "5. Error handling should be robust, capturing and logging exceptions where necessary.\n"
            "6. The code should be modular and reusable.\n"
            "7. If external libraries are used, they should be commonly used and well-maintained.\n"
            "8. The code should be directly related to the following context:\n"
        )
        messages = [
            {"role": "system", "content": rules},
            {"role": "user", "content": context}
        ]
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=messages
        )
        return response['choices'][0]['message']['content']

    @eel.expose
    async def identify_placeholders(self, code_str):
        try:
            # Debugging line to check the received code_str
            print(f"Debug: Received code_str = {code_str}")

            # Call the identify_placeholders_with_gpt4 method and get the values
            identified_placeholders = await self.identify_placeholders_with_gpt4(code_str)
        
            # Create a dictionary to store line numbers where placeholders are identified
            line_numbers_dict = {}
        
            for placeholder, line_num in identified_placeholders.items():
                line_numbers_dict[placeholder] = line_num

            # Return the dictionary
            return line_numbers_dict
        
        except Exception as e:
            # Log the exception for debugging
            print(f"An error occurred while identifying placeholders: {e}")
        
            # Return an empty dictionary to indicate that no placeholders were identified
            return {}

    
    @eel.expose
    def set_openai_api_key(self, api_key):
        openai.api_key = api_key
        # Save the API key to config.json
        with open("config.json", "r+") as f:
            config = json.load(f)
            config["openai_api_key"] = api_key
            f.seek(0)
            json.dump(config, f)
            f.truncate()

    @eel.expose
    def set_weaviate_client_url(self, client_url):
        # Update the client URL in the Python class
        manager.client = Client(client_url)
    
        # Save the client URL to config.json
        with open("config.json", "r+") as f:
            config = json.load(f)
            config["weaviate_client_url"] = client_url
            f.seek(0)
            json.dump(config, f)
            f.truncate()

    @eel.expose
    async def fill_placeholders(self, code_str):
        placeholder_lines = await self.identify_placeholders_with_gpt4(code_str)
        lines = code_str.split('\n')
        for placeholder, line_num in placeholder_lines.items():
            context = '\n'.join(lines[max(0, line_num-5):line_num])
            quantum_id = self.generate_quantum_id(context)
            
            # Quantum data storage
            self.store_data_in_weaviate("QuantumData", {"quantumID": str(quantum_id)})
            
            # Retrieve relevant code from Weaviate
            relevant_code = self.retrieve_relevant_code_from_weaviate(quantum_id)
            
            if relevant_code:
                new_code = relevant_code
            else:
                new_code = await self.generate_code_with_gpt4(context)

            lines[line_num] = f"{new_code}  # Quantum ID: {quantum_id}"
            self.store_data_in_weaviate("CodeSnippet", {"code": new_code, "quantumID": str(quantum_id)})
        
        return '\n'.join(lines)

async def run_asyncio_tasks(manager):
    while True:
        await asyncio.sleep(1)

async def start_eel():
    eel.init('web')
    manager = QuantumCodeManager()

    # Suggest better quantum circuit logic using GPT-4
    suggested_logic = await manager.suggest_quantum_circuit_logic()
    print(f"Suggested Quantum Circuit Logic:\n{suggested_logic}")

    eel.start('index.html', size=(720, 1280))
    return manager  # Return the manager object

if __name__ == "__main__":
    # Start Eel and get the manager object
    manager = asyncio.run(start_eel())

    # Run asyncio event loop in the main thread
    asyncio.run(run_asyncio_tasks(manager))
