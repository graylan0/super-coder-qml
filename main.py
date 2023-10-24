import eel
import openai
import aiohttp
import numpy as np
import pennylane as qml
import asyncio
import hashlib
import traceback
import json
from typing import Callable
from weaviate.util import generate_uuid5
from weaviate import Client
from transformers import pipeline
import requests
from PIL import Image, ImageTk
import io
import base64
import random
import logging
import sys

def normalize(value, min_value, max_value):
    """Normalize a value to a given range."""
    return min_value + (max_value - min_value) * (value / 0xFFFFFFFFFFFFFFFF)

class AestheticEvaluator:
    def __init__(self):
        self.pipe = pipeline("image-classification", model="cafeai/cafe_aesthetic")

    def evaluate_aesthetic(self, image_path):
        result = self.pipe(image_path)
        return result[0]['score']

class QuantumCodeManager:
    def __init__(self):
        self.aesthetic_evaluator = AestheticEvaluator()
        self.circuit_vector = []  # Initialize an empty list to store circuits

        # Load settings from config.json
        with open("config.json", "r") as f:
            config = json.load(f)
        try:
            self.openai_api_key = config["openai_api_key"]
        except KeyError:
            print("openai_api_key not found in config.json")
            weaviate_client_url = config.get("weaviate_client_url", "http://localhost:8080")

        self.session = aiohttp.ClientSession()  # Create an aiohttp session      

        # Initialize OpenAI API key
        openai.api_key = self.openai_api_key  # Consider using OpenAI's official method if available

        self.client = Client(weaviate_client_url)
        self.dev = qml.device("default.qubit", wires=2)  # Define self.dev here

        # Now it's safe to call this
        self.set_quantum_circuit(self.default_quantum_circuit)  # Set the default circuit and add it to the vector

    def __del__(self):
        self.session.close()
        
    def set_quantum_circuit(self, circuit_func: Callable):
        """Set a new quantum circuit function and apply the QNode decorator."""
        self.quantum_circuit = qml.qnode(self.dev)(circuit_func)
        self.circuit_vector.append(circuit_func)  # Append the new circuit to the vector

    def generate_images(self, message):
        url = 'http://127.0.0.1:7860/sdapi/v1/txt2img'
        payload = {
            "prompt": message,
            "steps": 50,
            "seed": random.randrange(sys.maxsize),
            "enable_hr": "false",
            "denoising_strength": "0.7",
            "cfg_scale": "7",
            "width": 1280,
            "height": 512,
            "restore_faces": "true",
        }
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            try:
                r = response.json()
                for i in r['images']:
                    image = Image.open(io.BytesIO(base64.b64decode(i.split(",", 1)[0])))
                    img_tk = ImageTk.PhotoImage(image)
                    eel.display_image(img_tk)  # Display the image in the GUI
            except ValueError as e:
                print("Error processing image data: ", e)
                logging.error(f"Error processing image data: {e}")  # Added logging
        else:
            print("Error generating image: ", response.status_code)

    def default_quantum_circuit(self, param1, param2):
        """Default quantum circuit definition."""
        qml.RX(param1, wires=0)
        qml.RY(param2, wires=1)
        qml.RZ(param1 + param2, wires=0)
        qml.CNOT(wires=[0, 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(2)]


    @eel.expose
    async def inject_data_into_weaviate(self, data: str):
        """
        Inject data into the Weaviate database.
        """
        # Generate a unique identifier for the data
        unique_id = generate_uuid5(data)

        # Create the data object in Weaviate
        try:
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
                if response.status != 200:
                    raise Exception(f"Failed to inject data: {await response.text()}")
                else:
                    print(f"Successfully injected data with ID: {unique_id}")
        except Exception as e:
            print(f"Error injecting data into Weaviate: {e}")
            raise e


    async def suggest_quantum_circuit_logic(self):
        """Use GPT-4 to suggest better logic for the quantum circuit."""
        # Get the last circuit in the vector for reference
        last_circuit = self.circuit_vector[-1] if self.circuit_vector else None
        prompt = f"Suggest a better logic for a quantum circuit aimed at solving optimization problems. The circuit must follow the Pennylane library. The last circuit used was: {last_circuit}"
        
        suggested_logic = await self.generate_code_with_gpt4(prompt)
        
        # Assuming the suggested_logic is a Callable, you can set it as the new circuit
        # self.set_quantum_circuit(suggested_logic)
        
        return suggested_logic.strip()

    async def optimize_code_with_llm(self, line):
        """Optimize a line of code using LLM (Language Model)."""
        prompt = f"Optimize the following line of code:\n{line}"
        optimized_line = await self.generate_code_with_gpt4(prompt)
        return optimized_line.strip()
    
    async def should_entangle(self, line):
        """Use LLM to decide if this line should be entangled with another line."""
        prompt = f"Should the following line of code be entangled with another line? If yes, provide the Quantum ID or context that it should be entangled with.\nLine: {line}"
        entanglement_decision = await self.generate_code_with_gpt4(prompt)
    
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

    async def execute_and_test_code(self, code_str):
        try:
            exec(code_str)
            return None  # No bugs
        except Exception as e:
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
        error_message, traceback_str = self.execute_and_test_code(code_str)
        
        if error_message:
            # Log the bug in Weaviate
            self.log_bug_in_weaviate(error_message, traceback_str, code_str)
            
            # Step 2: Use GPT-4 to suggest a fix
            suggested_fix = await self.generate_code_with_gpt4(f"Fix the following bug:\n{error_message}\n\nIn the code:\n{code_str}")
            
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

    async def store_data_in_weaviate(self, class_name, data):
        try:
            # Generate a deterministic UUID based on the data
            unique_id = generate_uuid5(data)
            
            # Create the data object in Weaviate asynchronously
            async with self.session.post(
                f"{self.client.url}/objects",
                json={
                    "class": class_name,
                    "id": unique_id,
                    "properties": data
                }
            ) as response:
                if response.status != 200:
                    print(f"Failed to store data: {await response.text()}")
        except Exception as e:
            print(f"Error storing data in Weaviate: {e}")

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
        # Assuming you have the OpenAI API set up
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{"role": "system", "content": f"Identify placeholders in the following Python code: {code_str}"}]
        )
        identified_placeholders = response['choices'][0]['message']['content'].split('\n')
        lines = code_str.split('\n')
        return {ph: i for i, line in enumerate(lines) for ph in identified_placeholders if ph in line}

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
            f"{context}\n\n"
            "Additional Information:\n"
            "- This application uses GPT-4 to identify and fill code placeholders.\n"
            "- A unique Quantum ID is generated for each code snippet, which is used for future reference and retrieval."
        )
        response = openai.ChatCompletion.create(
            model='gpt-4',
            messages=[{"role": "system", "content": rules}]
        )
        return response['choices'][0]['message']['content']

    @eel.expose
    async def identify_placeholders(self, code_str):
        return list((await self.identify_placeholders_with_gpt4(code_str)).values())
    
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

def run_asyncio_tasks(manager):
    loop = asyncio.get_event_loop()
    while True:
        loop.run_until_complete(asyncio.sleep(1))

def start_eel():
    eel.init('web')
    manager = QuantumCodeManager()

    # Suggest better quantum circuit logic using GPT-4
    loop = asyncio.get_event_loop()
    suggested_logic = loop.run_until_complete(manager.suggest_quantum_circuit_logic())
    print(f"Suggested Quantum Circuit Logic:\n{suggested_logic}")

    eel.start('index.html', block=True)
    return manager  # Return the manager object

if __name__ == "__main__":
    # Start Eel and get the manager object
    manager = start_eel()

    # Run asyncio event loop in the main thread
    run_asyncio_tasks(manager)
