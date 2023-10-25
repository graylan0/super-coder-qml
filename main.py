import json
import asyncio
import logging
import aiosqlite  
from concurrent.futures import ThreadPoolExecutor
from twitchio.ext import commands
from llama_cpp import Llama  # Assuming you have this package installed
import textwrap  # Add this line to import the textwrap module
import eel
from html import escape  # Import the escape function for sanitization
import openai
import aiohttp
import numpy as np
import pennylane as qml
import hashlib
import traceback
from typing import Callable
from types import FunctionType
from weaviate.util import generate_uuid5
from weaviate import Client

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Initialize twitch_token and initial_channels with default values
twitch_token = None
initial_channels = ["freedomdao"]

# Load configuration from config.json
try:
    with open("config.json", "r") as f:
        config = json.load(f)
    twitch_token = config.get("TWITCH_TOKEN")
    initial_channels = config.get("INITIAL_CHANNELS", ["freedomdao"])
except Exception as e:
    logging.error(f"Failed to load config.json: {e}")

# Initialize Llama model
llm = Llama(
  model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
  n_gpu_layers=-1,
  n_ctx=3900,
)

executor = ThreadPoolExecutor(max_workers=3)

async def run_llm(prompt):
    return await asyncio.get_event_loop().run_in_executor(executor, lambda: llm(prompt, max_tokens=900)['choices'][0]['text'])

# New function for GPT-4 (Added)
async def run_gpt4(prompt):
    return await asyncio.get_event_loop().run_in_executor(executor, lambda: openai.ChatCompletion.create(
        model='gpt-4',
        messages=[{"role": "user", "content": prompt}]
    )['choices'][0]['message']['content'])


# TwitchBot class definition
class TwitchBot(commands.Bot):
    def __init__(self):
        super().__init__(token=twitch_token, prefix="!", initial_channels=initial_channels)

    async def event_message(self, message):
        await self.handle_commands(message)
        await self.send_message_to_gui(message)  # Add this line to send messages to the GUI
        await self.save_chat_message_to_db(message.content)  # Save chat message to DB

    async def save_chat_message_to_db(self, message):
        async with aiosqlite.connect("chat_messages.db") as db:
            await db.execute("CREATE TABLE IF NOT EXISTS chat_messages (message TEXT)")
            await db.execute("INSERT INTO chat_messages (message) VALUES (?)", (message,))
            await db.commit()

    async def get_all_chat_messages_from_db(self):
        async with aiosqlite.connect("chat_messages.db") as db:
            cursor = await db.execute("SELECT message FROM chat_messages")
            return [row[0] for row in await cursor.fetchall()]

    async def event_ready(self):
        print(f'Logged in as | {self.nick}')

    async def event_message(self, message):
        await self.handle_commands(message)
        await self.send_message_to_gui(message)  # Add this line to send messages to the GUI

    @commands.command(name="llama")
    async def llama_command(self, ctx):
        prompt = ctx.message.content.replace("!llama ", "")
        reply = await run_llm(prompt)

        # Split the reply into chunks of 500 characters
        chunks = textwrap.wrap(reply, 500)

        # Send each chunk as a separate message
        for chunk in chunks:
            await ctx.send(chunk)

    async def send_message_to_gui(self, message):
        """Send sanitized Twitch chat messages to the Eel GUI."""
        sanitized_message = escape(message.content)  # Sanitize the message
        eel.receive_twitch_message(sanitized_message)  # Assuming you have a receive_twitch_message function in your Eel JavaScript

def normalize(value, min_value, max_value):
    """Normalize a value to a given range."""
    return min_value + (max_value - min_value) * (value / 0xFFFFFFFFFFFFFFFF)

class QuantumCodeManager:
    def __init__(self):
        self.circuit_vector = []  # Initialize an empty list to store circuits

        # Load settings from config.json
        try:
            with open("config.json", "r") as f:
                config = json.load(f)
            self.openai_api_key = config["openai_api_key"]
            weaviate_client_url = config.get("weaviate_client_url", "http://localhost:8080")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            print(f"Error reading config.json: {e}")
            self.openai_api_key = None
            weaviate_client_url = "http://localhost:8080"

        # Initialize Weaviate client
        self.client = Client(weaviate_client_url)

        # Initialize aiohttp session
        self.session = aiohttp.ClientSession()

        # Initialize OpenAI API key
        if self.openai_api_key:
            openai.api_key = self.openai_api_key  # Consider using OpenAI's official method if available

        # Initialize quantum device
        self.dev = qml.device("default.qubit", wires=2)

        # Set the default quantum circuit
        self.set_quantum_circuit(self.default_quantum_circuit)

    def __del__(self):
        self.session.close()

        # Initialize SQLite database
        self.db_path = "prompts.db"
        loop = asyncio.get_event_loop()
        loop.run_until_complete(self.init_db())
        
    async def init_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("CREATE TABLE IF NOT EXISTS prompts (prompt TEXT)")
            await db.commit()

    async def save_prompt_to_db(self, prompt):
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("INSERT INTO prompts (prompt) VALUES (?)", (prompt,))
            await db.commit()

    async def get_all_prompts_from_db(self):
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT prompt FROM prompts")
            return [row[0] for row in await cursor.fetchall()]

    async def fetch_data_from_db(self):
        data_list = []
        async with aiosqlite.connect("prompts.db") as db:
            cursor = await db.execute("SELECT * FROM your_table")
            async for row in cursor:
                data_list.append(row)
        return data_list

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


    @eel.expose
    async def inject_data_into_weaviate(self, prompt: str, timestamp: str, user_reply: str):
        """
        Inject data into the Weaviate database.
        """
        # Generate a unique identifier for the data
        unique_id = generate_uuid5(prompt)

        # Create the data object in Weaviate
        try:
            async with self.session.post(
                f"{self.client.url}/objects",
                json={
                    "class": "InjectedData",
                    "id": unique_id,
                    "properties": {
                        "prompt": prompt,
                        "timestamp": timestamp,
                        "user_reply": user_reply
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
    
        if not suggested_logic:
            print("Error: GPT-4 did not return any suggested logic.")
            return None

        # Try to convert the string to a callable function
        try:
            exec_globals = {}
            exec_locals = {}
            exec(f"{suggested_logic.strip()}", exec_globals, exec_locals)
        
            # Find the first callable in the local namespace and set it as the new circuit
            for name, obj in exec_locals.items():
                if isinstance(obj, FunctionType):
                    self.set_quantum_circuit(obj)
                    break
            else:
                print("Error: No callable function found in the suggested logic.")
                return None

        except Exception as e:
            print(f"Error: Could not convert the suggested logic to a callable function. Exception: {e}")
            return None
    
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
    # New function to run both models (Added)
    async def run_both_models(prompt):
        gpt4_output = await run_gpt4(prompt)
        llama_output = await run_llm(prompt)
        return gpt4_output + "\n" + llama_output

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
    
# New function to retrieve all chat messages from SQLite database
async def get_all_chat_messages_from_db():
    async with aiosqlite.connect("chat_messages.db") as db:
        cursor = await db.execute("SELECT message FROM chat")
        return [row[0] for row in await cursor.fetchall()]
    
# Expose the function to Eel
@eel.expose
async def fetch_chat_messages():
    return await get_all_chat_messages_from_db()

async def execute_tasks(manager, tasks):
    """Execute a list of tasks concurrently and return their results."""
    return await asyncio.gather(*tasks)

async def send_prompts_to_twitch(bot, prompts):
    """Send replies for a list of prompts to Twitch."""
    for prompt in prompts:
        reply = await run_llm(prompt)
        await bot.send_message("freedomdao", f"Reply for {prompt}: {reply}")
        # Save the prompt to the database
        await manager.save_prompt_to_db(prompt)

async def run_asyncio_tasks(manager, bot):
    while True:
        # Fetch data from the database
        data_list = await manager.fetch_data_from_db()
        
        # Create a list of tasks for data injection into Weaviate
        weaviate_tasks = [manager.inject_data_into_weaviate(data['prompt'], data['timestamp'], data['user_reply']) for data in data_list]
        
        # Execute tasks and wait for their completion
        weaviate_results = await asyncio.gather(*weaviate_tasks)
        
        # Notify on Twitch when all tasks for Weaviate are completed
        await bot.send_message("freedomdao", "All tasks for Weaviate have been completed.")
        
        # Get all prompts from the database
        prompts = await manager.get_all_prompts_from_db()
        
        # Create a list of tasks for sending prompts to Twitch
        twitch_tasks = [send_prompts_to_twitch(bot, prompt) for prompt in prompts]
        
        # Execute tasks and wait for their completion
        twitch_results = await asyncio.gather(*twitch_tasks)

        # Log or handle the results
        print(f"Twitch tasks results: {twitch_results}")

        # Notify on Twitch when all tasks for sending prompts are completed
        await bot.send_message("freedomdao", "All tasks for sending prompts to Twitch have been completed.")

def start_eel():
    eel.init('web')
    manager = QuantumCodeManager()

    # Suggest better quantum circuit logic using GPT-4
    loop = asyncio.get_event_loop()
    suggested_logic = loop.run_until_complete(manager.suggest_quantum_circuit_logic())
    print(f"Suggested Quantum Circuit Logic:\n{suggested_logic}")

    eel.start('index.html', block=True)
    return manager  # Return the manager object

# Main function
if __name__ == "__main__":
    import nest_asyncio
    nest_asyncio.apply()

    # Initialize Twitch bot
    bot = TwitchBot()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(bot.run())

    # Initialize Eel and QuantumCodeManager
    eel.init('web')
    manager = QuantumCodeManager()

    # Start the asyncio tasks
    loop.run_until_complete(run_asyncio_tasks(manager, bot))

    # Start Eel
    eel.start('index.html', block=True)
