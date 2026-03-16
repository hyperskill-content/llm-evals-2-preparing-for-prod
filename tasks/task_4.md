## Managing API Keys and Budgets via LiteLLM Proxy

### **Table of Contents**

- [Description](#description)
- [Development Steps](#development-steps)
- [Deliverables](#deliverables)
- [Useful Resources](#useful-resources)
    - [Docs](#docs)

---

### Description

Once you get your API keys from your model provider, it is a good idea for your applications to use them via an LLM proxy server. This allows you to perform additional actions before the requests are routed to your model provider. For instance, you can enforce budget constraints, set rate limits, add load balancing, caching, and more.

[LiteLLM](https://github.com/BerriAI/litellm) is an open-source LLM gateway that serves as a control center for your API keys. Key benefits include:

- Standardized access: A unified API that works with multiple LLM providers.
- Intelligent load balancing: Smart traffic distribution using least-busy, round-robin, and weighted routing.
- Fault tolerance: Built-in recovery with automatic retries and model failover options.
- Centralized management: Single dashboard for API keys, logging, cost tracking, and monitoring.
- Cost optimization: Smart request routing based on cost and performance metrics.

In this task, you will set up and use the [LiteLLM proxy](https://docs.litellm.ai/) to manage your API key. Through a set of configs, we will create virtual keys that we can assign to applications or users, which in turn allow us to enforce certain constraints, such as user budgets and rate limits.

---

### Development Steps

There are various ways we can deploy and use the LiteLLM proxy. For all methods, check out the [setup and deployment guides](https://docs.litellm.ai/docs/proxy/deploy). We recommend deploying via Docker Compose. First, you need to [clone the repository](https://github.com/BerriAI/litellm.git).

LiteLLM is configured via a `config.yaml` file. So, you need to create a new `config.yaml` file in the root directory of the project you just cloned, relative to the `docker-compose.yml` file. In that file, you’d add the appropriate configs for the models we have in the project (for example, the embeddings and chat model). Here is a [production-ready](https://docs.litellm.ai/docs/proxy/prod) template you can use for your `config.yaml`:

```yaml
model_list:
  - model_name: <model_name> # this is the model name your application will use for instance the chat model
    litellm_params: 
      model: <actual_model_name> # the actual model that LiteLLM will call
      api_base: <actual_base_url> # the actual base URL for your model provider
      api_key: os.environ/<actual_api_key> # does os.getenv("actual_api_key") to get your actual API key
      rpm: 10 # Rate limit for this deployment: in requests per minute (rpm)
      tpm: 500 # token limit for this deployment: in tokens per minute (tpm)
      num_retries: 3
      
  - model_name: <another_model> # another model for you application for example the embeddings model
    litellm_params:
      model: <another_actual_model_name> # model name that LiteLLM will call
      api_key: os.environ/<another_actual_api_key> # does os.getenv("another_actual_api_key") to get your actual API key
      num_retries: 5 
      mode: embedding
litellm_settings: # module level litellm settings - https://github.com/BerriAI/litellm/blob/main/litellm/__init__.py
  drop_params: True
  success_callback: ["langfuse"] # OPTIONAL - if you want to start sending LLM Logs to Langfuse. Make sure to set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` in your env
  max_budget: 5 # maximum budget for the models in cents
  max_end_user_budget: 0.002 # maximum budget for end users. Applies if you pass in "user" when making calls
  budget_duration: 1d # how long before the budget resets. Here we use 1 day but you can also use seconds, minutes, hours, days (1s, 1m, 1h, 1d) etc
  request_timeout: 600 # raise Timeout error if call takes longer than 600 seconds. Default value is 6000seconds if not set
  set_verbose: False # Switch off Debug Logging, ensure your logs do not have any debugging on
  json_logs: true # debug logs in json format
general_settings:
  master_key: sk-deploying-llm-applications # the master key the app will use. must start with sk
  salt_key: deploying-llm-apps # used to encrypt and decrypt your API key. Use https://1password.com/password-generator to generate a good one
  database_url: "postgresql://llmproxy:dbpassword9090@db:5432/litellm" 
  
```

For a full list of options that you can configure in your `config.yaml` file, check [out the docs](https://docs.litellm.ai/docs/proxy/config_settings), but these should be enough for our use case. Once the proxy is up and running, we will use the admin UI to configure additional settings. Feel free to adjust the values for what works well for you. Also, experiment with adding options such as [alerting](https://docs.litellm.ai/docs/proxy/alerting), [routing and fallbacks](https://docs.litellm.ai/docs/routing-load-balancing), and even [caching](https://docs.litellm.ai/docs/proxy/caching).

With the `config.yaml` file in place, we also need to slightly modify the `docker-compose.yml` file to ensure that it uses this configuration file. Look for the volumes option and add the following to map the local `config.yaml` file you created to the container and pass it to the start command:

```yaml
volumes:
  - ./config.yml:/app/config.yml
command: [ "--config", "/app/config.yml", "--port", "4000"]
```

Ensure that you create and map the file as *config.yaml,* not *config.yml*. Also, if the PostgreSQL or Prometheus ports are already in use, you need to map them to different host ports. (For example, port `5432` for PostgreSQL is already being used by Langfuse. The syntax is `*host port:container port`.*)

Now, you can start the LiteLLM proxy with `docker compose up -d` (option -d runs it in the background, but you can omit it for debugging since it may take a long time to start). Then, verify that it is available at [localhost](http://0.0.0.0:4000/v1/models). Once you have the proxy up and running, you just need to head over to the [LiteLLM admin UI](http://0.0.0.0:4000/ui/). To log in, you use the username `admin` and the master key you set in `config.yaml` as the password. Here is what the dashboard looks like:

![LiteLLM Dashboard](../assets/images/litellm_dashboard.png)

Here, you can generate virtual keys for users. This allows you to track the user’s current usage and rate limit/reject/route their requests.  You can also use `curl` to create virtual keys for your users or teams. Here’s how you might create a virtual key (replace <your-master-key>) and set the models you’d like the key to be able to access:

```bash
curl 'http://0.0.0.0:4000/key/generate' \
--header 'Authorization: Bearer <your-master-key>' \
--header 'Content-Type: application/json' \
--data-raw '{"models": ["gpt-4o-mini", "gpt-4"], "max_budget": 0.00100, "budget_duration": "2m"}'
```

This virtual key now has a set budget of $0.001, which will reset every two minutes. Place the new virtual key in your `.env` file, with [`http://0.0.0.0:4000`](http://0.0.0.0:4000) as the base URL.

Another useful way to set up budgets is to create one in the UI. For example, you might set up a budget `HyperBudget`, that will apply to all end users. You can create different budgets for different pricing tiers and assign them to different users. Then, use `CURL` to create a new user and associate them with a budget via `budget_id` (depending on their tier):

```bash
curl -X POST 'http://0.0.0.0:4000/customer/new' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <your_master_key>' \
  -d '{
    "user_id": "HyperUser",
    "budget_id": "HyperBudget" # associate user with budget
  }'
```

> Such a request would be sent during user creation, so that every user who signs up on your platform is associated with a particular budget.

For this to apply, you need to pass “user” when calling LLMs. With LangChain, you might do it via `model_kwargs` as follows:

```python
llm = ChatOpenAI(
    model=os.getenv("LITELLM_MODEL"),
    base_url="http://0.0.0.0:4000/",
    api_key=os.getenv("LITELLM_API_KEY"),
    model_kwargs={"user": "HyperUser"},
)
```

You can also do it during invocation:

```python
response = llm.invoke(
    messages,
    user="HyperUser"
)
```
This way, LiteLLM will know which user is making the request and apply the budget and rate limits accordingly.
As an extra challenge, you can also try to use LiteLLM to set up [caching](https://docs.litellm.ai/docs/proxy/caching) and [routing](https://docs.litellm.ai/docs/proxy/provider_budget_routing) for your application. For example,you can set up routing so that requests to the chat model are routed to different providers based on the user’s budget or rate limits.

---

### Deliverables

You should now have the LiteLLM proxy running in your local environment. Your application should use this proxy to manage API keys, budget, and rate/token limits. This means using virtual keys and adding a “user” attribute to your LangChain LLM client or chains via `model_kwargs`.

---

### **Useful Resources**

###### **Topics**

- [Overview of Prometheus](https://hyperskill.org/learn/step/52690).

###### **Docs**

- [LiteLLM Repo](https://github.com/BerriAI/litellm.git).
- [LiteLLM Setup and Deployment Guide](https://docs.litellm.ai/docs/proxy/deploy).
- [LiteLLM Getting Started Guide](https://docs.litellm.ai/docs/proxy/docker_quick_start).
- [LiteLLM Admin UI](https://docs.litellm.ai/docs/proxy/ui).
- [Budgets and Rate Limits](https://docs.litellm.ai/docs/proxy/users).
- [Virtual Keys](https://docs.litellm.ai/docs/proxy/virtual_keys).
- [Setting Customer Budgets and Pricing Tiers](https://docs.litellm.ai/docs/proxy/customers#setting-customer-budgets).

###### **Other useful articles**

- [LangChain ChatLiteLLM and ChatLiteLLMRouter](https://python.langchain.com/docs/integrations/chat/litellm/).




