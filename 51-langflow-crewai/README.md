1. What is langflow 

Langflow is a new visual way to build ,iterate and deploy AI application 


Setup processus : 

# Make sure you have Python 3.10 installed on your system.
# Install the pre-release version
python -m pip install langflow --pre --force-reinstall

# or stable version

python -m pip install langflow -U


langflow commands :


Then, run Langflow with:

# python -m langflow run




You can run the Langflow using the following command:

langflow run [OPTIONS]
Each option is detailed below:

--help: Displays all available options.
--host: Defines the host to bind the server to. Can be set using the LANGFLOW_HOST environment variable. The default is 127.0.0.1.
--workers: Sets the number of worker processes. Can be set using the LANGFLOW_WORKERS environment variable. The default is 1.
--timeout: Sets the worker timeout in seconds. The default is 60.
--port: Sets the port to listen on. Can be set using the LANGFLOW_PORT environment variable. The default is 7860.
--env-file: Specifies the path to the .env file containing environment variables. The default is .env.
--log-level: Defines the logging level. Can be set using the LANGFLOW_LOG_LEVEL environment variable. The default is critical.
--components-path: Specifies the path to the directory containing custom components. Can be set using the LANGFLOW_COMPONENTS_PATH environment variable. The default is langflow/components.
--log-file: Specifies the path to the log file. Can be set using the LANGFLOW_LOG_FILE environment variable. The default is logs/langflow.log.
--cache: Selects the type of cache to use. Options are InMemoryCache and SQLiteCache. Can be set using the LANGFLOW_LANGCHAIN_CACHE environment variable. The default is SQLiteCache.
--dev/--no-dev: Toggles the development mode. The default is no-dev.
--path: Specifies the path to the frontend directory containing build files. This option is for development purposes only. Can be set using the LANGFLOW_FRONTEND_PATH environment variable.
--open-browser/--no-open-browser: Toggles the option to open the browser after starting the server. Can be set using the LANGFLOW_OPEN_BROWSER environment variable. The default is open-browser.
--remove-api-keys/--no-remove-api-keys: Toggles the option to remove API keys from the projects saved in the database. Can be set using the LANGFLOW_REMOVE_API_KEYS environment variable. The default is no-remove-api-keys.
--install-completion [bash|zsh|fish|powershell|pwsh]: Installs completion for the specified shell.
--show-completion [bash|zsh|fish|powershell|pwsh]: Shows completion for the specified shell, allowing you to copy it or customize the installation.
--backend-only: This parameter, with a default value of False, allows running only the backend server without the frontend. It can also be set using the LANGFLOW_BACKEND_ONLY environment variable.
--store: This parameter, with a default value of True, enables the store features, use --no-store to deactivate it. It can be configured using the LANGFLOW_STORE environment variable.
