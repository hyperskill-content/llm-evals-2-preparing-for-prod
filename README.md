## Table of Contents

- [Introduction](#introduction)
- [Learning Outcomes](#learning-outcomes)
- [Project Structure](#project-structure)
- [Tasks](#tasks)
    - [Task One — Prompt Management and Versioning](./tasks/task_1.md)
    - [Task Two — Chat History Management](./tasks/task_2.md)
    - [Task Three — LLM Safety and Security: Guardrails](./tasks/task_3.md)
    - [Task Four — Managing API Keys and Budgets via LiteLLM Proxy.](./tasks/task_4.md)
- [Deliverables](#deliverables)
- [Useful Resources](#useful-resources)
- [Contributing](#contributing)

## Introduction

In the [first part](https://github.com/hyperskill-content/LLM-evals) of this series, we established the foundation: a complete evaluation pipeline. We now have the tools and metrics to understand our application's performance, identify hallucinations, and measure the quality of its responses. We know what makes our smartphone info bot “good,” but a successful application needs more than just good performance — it needs to be reliable, safe, and efficient in a live environment.

A prototype application often has hidden issues. Its prompts might be hard-coded, making them difficult to update or A/B test. It might slow down or become costly with repeated similar queries. Without proper safeguards, it could be tricked into going off-topic or responding in an inappropriate manner. As the conversation history grows, it can lose context or exceed token limits, which breaks the user experience. These are the challenges you’ll face when moving from a proof-of-concept to a production-ready system.

In this project, we will tackle these challenges head-on. We will upgrade the core architecture of our chatbot to enhance its robustness. We'll replace static prompts with a versioned management system, implement proper chat history management, ensure cost efficiency, and build a safety net around our LLM with programmable guardrails.

By focusing on these production-oriented patterns, we are building a resilient application that will be ready for scalable deployment in the upcoming project.

---

## **Setup**

This project uses [uv](https://docs.astral.sh/uv/) for Python package management. Follow these steps to get started:

### Prerequisites

1. Install uv:
   ```bash
   # On macOS and Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # On Windows
   powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Verify installation:
   ```bash
   uv --version
   ```

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm-evals-2-preparing-for-prod
   ```

2. Install dependencies using `uv`:
   ```bash
   uv sync
   ```

   This will:
   - Create a virtual environment in `.venv/`
   - Install all dependencies from `pyproject.toml`
   - Generate/update the `uv.lock` file for reproducible builds

3. Set up environment variables:
   ```bash
   cp .env.sample .env
   # Edit .env with your API keys and configuration
   ```

### Running the Application

```bash
uv run python main.py
```

If your IDE (VS Code, PyCharm) isn't recognizing the environment or you need to use tools that aren't uv-aware, activate the virtual environment manually:
```bash
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate  # On Windows

# Run the application
python main.py
```

### Adding New Dependencies

To add new packages to the project:

```bash
uv add <package-name>
```

For development dependencies:

```bash
uv add --dev <package-name>
```

### Managing Environment Variables

When adding new environment variables to your project:

1. **Always update `.env.sample`** with the new variable names (without actual values):
   ```bash
   # Example: Adding a new service
   NEW_SERVICE_API_KEY="<your-api-key-here>"
   NEW_SERVICE_BASE_URL="<service-url>"
   ```

2. **Do not remove `.env.sample`** - This file serves as a template for other developers and documents all required environment variables for the project.

3. **Add descriptive comments** in `.env.sample` to explain what each variable is used for, especially if it's not immediately obvious.

### Documenting Code Structure Changes

If you modify the default application structure or change how the application runs:

1. **Update this README** with the new run instructions in the "Running the Application" section above.

2. **Document any new directories or files** in the "Project Structure" section.

3. **Explain the reasoning** for structural changes in your Pull Request description so reviewers understand the architectural decisions.

Examples of changes that require documentation updates:
- Moving code from `main.py` into modular files under `src/`
- Introducing new entry points (e.g., `app.py`, `cli.py`)
- Adding configuration files or changing how configuration is loaded
- Creating new directories for components, utilities, or services       

## **Learning Outcomes**

By the end of this project, you will have transformed the functional chatbot prototype into a production-ready application. You'll implement key operational patterns that ensure reliability and control. This project will equip you with the skills to build LLM applications that are secure and efficient — ready to handle real-world deployment challenges.

---

## **Project Structure**

Here are the main directories and files in this repo:

```markdown
├── images/
│   ├── litellm_dashboard.png
│   ├── new_prompt.png
│   ├── prompt_links.png
│   ├── prompt_observations.png
│   └── prompt_playgrounds.png
├── tasks/
│   ├── task_1.md
│   ├── task_2.md
│   ├── task_3.md
│   ├── task_4.md
│   └── task_5.md
├── .env.sample
├── .gitignore
├── CONTRIBUTING.md
├── main.py
├── README.md
└── requirements.txt
```

## **Tasks**

The project is divided into various tasks that you need to complete. The tasks are located in the [tasks folder](./tasks) of the repository. Each task includes all the necessary objectives, suggested development steps, deliverables, and useful resources. Here's a quick primer on each task:

- **Task One — Prompt management and versioning:** Implement version control for prompts, allowing for easier updates, tracking, and experimentation without changing application code.
- **Task Two — Chat history management:** Design robust strategies for managing chat history, ensuring the chatbot can handle long, context-rich conversations without failure.
- **Task Three — Managing API Keys and Budgets via LiteLLM Proxy:** Use LiteLLM to manage your API keys, set budgets per user per key, enforce rate limits, and more.
- **Task Four — LLM safety and security:** Implement programmable safeguards using NVIDIA's NeMo Guardrails to control the chatbot's conversational boundaries, prevent topical deviations, and ensure it responds safely and appropriately.
---

## **Useful Resources**

Each task contains a collection of resources that will be helpful for you as you solve the task. There are links to topics and projects, documentation, and other helpful tutorials that you can use. You may not always need to use all the provided resources if you're already familiar with the concepts. In addition to the provided resources, you can always discuss with others and experts. You can use various channels — GitHub Issues, GitHub Discussions, PRs, or Discord.

---

## **Deliverables**

Each task contains a set of deliverables that bring you close to achieving the final goal. The final product is a production-ready LLM application.

---

## **The Flow**

Fork → Clone → Branch → Implement → PR → Review

- Fork this repo to your own GitHub account.
- Create a new branch for each task (e.g., task-1) if applicable (if there is any code that has to be implemented).
- Implement the solution based on the task descriptions.
- Push the branch to the forked repo.
- Create a Pull Request from the fork back to the main repo.
- We will review the PR and provide feedback through GitHub.

Next: [Prompt Management and Versioning](./tasks/task_1.md)
