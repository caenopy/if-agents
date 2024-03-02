# if-agents

### Requirements

Install `jericho`, `dspy-ai`, `tqdm`.

Install the Google GenerativeAI SDK to use Gemini models: `pip install -q -U google-generativeai`.

### Setup

#### Acquire games
```
mkdir data
cd data
wget https://github.com/BYU-PCCL/z-machine-games/archive/master.zip
unzip master.zip
```

### LLM API Keys

Create environmental variables for each language model API:

OpenAI:

```
export OPENAI_API_KEY=<YOUR KEY HERE>
```

Together:

```
export TOGETHER_API_BASE="https://api.together.xyz/inference"
export TOGETHER_API_KEY=<YOUR KEY HERE>
```