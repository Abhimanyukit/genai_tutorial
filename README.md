# GenAI Tutorial 

## Install Required Packages

```
    pip install sentence-transformers

```

## Import the Model

```
    from  sentence_transformers import SentenceTransformer

```

# FAISS (Industry Standard)

## Install Packages 

```
    pip install faiss-cpu

```

## Install Required Packages

```
    pip install chromadb

```

## Install Required Packages to implement chromadb

```
   pip install chromadb sentence-transformers openai

```

## Install Required Packages to use ollama free on local server

```
   pip install ollama

```

## ollama --version

```

If you get something like:

ollama version 0.11.5

then it's installed.

If you get:

command not found: ollama

then install it from:

https://ollama.com/download

```

## Install Required Packages to use ollama free on local server

```
   curl -fsSL https://ollama.com/install.sh | sh

```

## Check if the Ollama server is running


```
   ollama list
```

## If you see something like:


```
   NAME          ID            SIZE
    llama3.2      xxxxx         2 GB
```

that's good.

## If instead you get a connection error, the server isn't running.

# Start the Ollama server

```
   ollama serve

   You should see output similar to:

    Listening on 127.0.0.1:11434

    Leave this terminal open.

    Open another terminal for running your Python program.
```

## Download a model


```
  Check available models:

    ollama list

    If nothing is listed:

    NAME    ID    SIZE

    download one:

    ollama pull llama3.2

    or

    ollama pull mistral
```

## Test the model


```
    Run:

    ollama run llama3.2

    You should see:

    >>>

    Type:

    What is Python?

    If it answers, Ollama is working correctly.

    Exit with:

    /bye
```

# Another Option: Hugging Face Transformers (Local)

```
    pip install transformers torch
```

##  Check the Ollama server is running 

```
http://localhost:11434

```


