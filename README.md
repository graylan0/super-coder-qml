# super-coder-qml EXPERIMENTAL (BUGS ARE NOT SOLVED)

![image](https://github.com/graylan0/super-coder-qml/assets/34530588/a0cf1373-465a-4f6e-bc84-3387d454e537)


Demo Image :  ![image](https://github.com/graylan0/super-coder-qml/assets/34530588/a944896b-0b50-48a4-b18a-0fa154669add)

![image](https://github.com/graylan0/super-coder-qml/assets/34530588/5a2227bf-d257-4fd4-9036-7968df4d4538)

An application that leverages quantum computing and machine learning to manage code snippets. It uses the PennyLane library to generate unique Quantum IDs for different code contexts and stores them in a Weaviate database. The application also utilizes GPT-4 to identify and fill code placeholders. It offers a Matrix-themed web interface via the Eel library, allowing users to interact with the application seamlessly. This project aims to provide a futuristic approach to code management and generation.

### Installation Guide



#### Windows

Install Weaviate by changing into the directory with the files after cloning the git (in step 6) by running  `docker compose up -d`  https://weaviate.io/developers/weaviate/installation/docker-compose

1. **Download Python 3.10**: Visit the [Python 3.10 Download Link](https://www.python.org/downloads/release/python-3100/) and download the installer for Windows.
    ```
    Click on "Windows installer (64-bit)"
    ```

2. **Install Python**: Run the downloaded installer.
    ```
    Double-click the installer -> Check "Add Python 3.10 to PATH" -> Install Now
    ```

3. **Download Git**: Visit the [Git Download Link](https://git-scm.com/download/win) and download the installer for Windows.
    ```
    Click on "Download for Windows"
    ```

4. **Install Git**: Run the downloaded installer.
    ```
    Double-click the installer -> Next -> Next -> ... -> Install
    ```

5. **Open Command Prompt**: Open the Run dialog.
    ```
    Press Win + R -> Type "cmd" -> Press Enter
    ```

6. **Clone Repository**:
    ```
    git clone https://github.com/graylan0/super-coder-qml
    ```

7. **Navigate to Project Directory**:
    ```
    cd super-coder-qml
    ```

8. **Install Required Libraries**: 
    ```
    pip install eel openai numpy pennylane weaviate-client asyncio
    ```
9. **Run the Program**: 
    ```
    python main.py
    ```




#### macOS

Install Weaviate by changing into the directory with the files after cloning the git (in step 6) by running  `docker compose up -d`  https://weaviate.io/developers/weaviate/installation/docker-compose

1. **Download Python 3.10**: Visit the [Python 3.10 Download Link](https://www.python.org/downloads/release/python-3100/) and download the macOS installer.
    ```
    Click on "macOS 64-bit universal2 installer"
    ```

2. **Install Python**: Open the downloaded `.pkg` file.
    ```
    Double-click the .pkg file -> Follow the installation instructions
    ```

3. **Download Git**: Open Terminal.
    ```
    Press Cmd + Space -> Type "Terminal" -> Press Enter
    ```

4. **Install Git**: 
    ```
    brew install git
    ```

5. **Clone Repository**:
    ```
    git clone https://github.com/graylan0/super-coder-qml
    ```

6. **Navigate to Project Directory**:
    ```
    cd super-coder-qml
    ```

7. **Install Required Libraries**: 
    ```
    pip3 install eel openai numpy pennylane weaviate-client asyncio
    ```
8. **Run the Program**: 
    ```
    python main.py
    ```


#### Ubuntu

Install Weaviate by changing into the directory with the files after cloning the git (in step 6) by running  `docker compose up -d`  https://weaviate.io/developers/weaviate/installation/docker-compose

1. **Update Repositories and Packages**: Before installing any software, it's good practice to update the repositories and installed packages to their latest versions.
    ```
    sudo apt update && sudo apt upgrade -y
    ```

2. **Install Dependencies for Adding PPA**: Install the necessary dependencies to add the repository.
    ```
    sudo apt install software-properties-common -y
    ```

3. **Add Deadsnakes PPA**: Add the deadsnakes PPA repository to get Python 3.10.
    ```
    sudo add-apt-repository ppa:deadsnakes/ppa
    ```

4. **Update Repositories Again**: Update the repositories once more to recognize the newly added PPA.
    ```
    sudo apt update
    ```

5. **Install Python 3.10**: Install Python 3.10 from the deadsnakes PPA.
    ```
    sudo apt install python3.10
    ```

6. **Install Git**: Install Git to clone the repository.
    ```
    sudo apt install git
    ```

7. **Clone Repository**: Clone the `super-coder-qml` repository from GitHub.
    ```
    git clone https://github.com/graylan0/super-coder-qml
    ```

8. **Navigate to Project Directory**: Change to the project directory.
    ```
    cd super-coder-qml
    ```

9. **Install Required Libraries**: Install all the Python libraries required for the project.
    ```
    pip3 install eel openai numpy pennylane weaviate-client asyncio
    ```
10. **Run the Program**: 
    ```
    python main.py
    ```
