# Homework 2 — Task 1: Human in the loop

This extension keeps the four original Task 6 roles. The orchestrator can now
route to `human` for clarification or approval. A LangGraph `interrupt` pauses
the graph and SQLite saves its state. The next command resumes the **same
thread ID** with `Command(resume=reply)`, and the reply is appended to the
shared dialogue as a message named `human`.

```text
user -> orchestrator -> scientist/general -> orchestrator
                    \-> human (clarify) -> orchestrator
                    \-> human (approve) -> executor -> orchestrator
                    \-> human (approve) -> finish
```

The executor runs only after explicit approval. A refusal returns to the
orchestrator, with a `system_status` message stating that the proposed action
did not run. The same approval rule applies before reporting the final
measurement. The previous Task 6 graph and scripts are unchanged.

## Setup

From the repository root, install `requirements.txt` in a virtual environment.
Create a local `.env` file with `GROQ_API_KEY=...` (do not commit it). Make
`task5_reference` point to the previous Task 5 repository as described in the
root README. The existing Model E checkpoint and processed NPZ file are used;
training is not repeated.

## Two required demonstrations

Run these in VS Code's terminal from the repository root, using the virtual
environment's Python. A start command prints the human question; use the
corresponding resume command after reading it. Keep each thread ID unchanged.

**A. Ambiguous request**

```bash
python -m homework2_task1.cli start --thread-id hw2-ambiguity-01 --request "Please explain the Silverbox record and then evaluate it. By record I might mean the data sample or the trained model."
python -m homework2_task1.cli resume --thread-id hw2-ambiguity-01 --reply "I mean the data sample: explain its input and output, then evaluate the trained model."
```

The first pause should be a clarification question. The transcript should
then show `human` followed by the next routed agent message. If the graph asks
for execution approval, answer with a new `resume` command.

**B. Refuse, then accept a hard decision**

```bash
python -m homework2_task1.cli start --thread-id hw2-approval-01 --request "Evaluate the saved Model E checkpoint on the Silverbox test segment and report the measured RMSE."
python -m homework2_task1.cli resume --thread-id hw2-approval-01 --reply "no"
python -m homework2_task1.cli resume --thread-id hw2-approval-01 --reply "yes"
```

After `no`, verify that the transcript says the step did not run and contains
no executor result. After `yes`, the executor may run for up to five minutes
per generated script. If asked to approve reporting the measured result,
resume the **same** thread with `yes` once more. The exact questions and
routing decisions are made by the model, so inspect each prompt before
answering. Do not start a new thread to continue an interrupted run.

Inspect or save the shared dialogue:

```bash
python -m homework2_task1.cli status --thread-id hw2-approval-01 --json-out outputs/homework2_task1/approval_transcript.json
```

The SQLite checkpoint and local transcripts stay under
`outputs/homework2_task1/` and are ignored by Git. For the report, copy only
the relevant verified turns and the actual printed RMSE. An offline check of
clarification, refusal, approval and persistent resume is available with:

```bash
python -m unittest homework2_task1.test_graph -v
```

That offline check uses test doubles; it does not constitute a Groq or
Silverbox execution result.
