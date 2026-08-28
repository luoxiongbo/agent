# SensorsAI Python SDK Skeleton

This project is a lightweight, dependency-free implementation of the SensorsAI
component architecture described in the product white paper.

## Component Model

- `MainAgent` is the only user-facing entry point.
- `SceneAgent` handles open-ended professional tasks delegated by `MainAgent`.
- `Skill` defines reusable capability contracts and may call tools or atomic agents.
- `Tool` connects the agent layer to concrete business capabilities.
- `ContextStore` keeps knowledge, memory, history, audit logs, and assets separate.
- `Governance` blocks actions when the user lacks required permission.

## Run Tests

```bash
python3 -m unittest discover -s tests -v
```

## Run Demo

```bash
python3 -m company_project.demo
```
