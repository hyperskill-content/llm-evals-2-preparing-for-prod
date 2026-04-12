from langfuse import Langfuse

YES_VOTES = {"yes", "y", "yeah", "yup", "sure", "absolutely", "of course"}
NO_VOTES = {"no", "n", "nope", "nop"}


def get_user_feedback(langfuse: Langfuse) -> None:
    feedback_value = None
    while feedback_value is None:
        feedback_input = input("Was this answer helpful? (Yes/No): ").strip().lower()
        if feedback_input in YES_VOTES:
            feedback_value = True
        elif feedback_input in NO_VOTES:
            feedback_value = False
        elif feedback_input in ["", "exit", "quit"]:
            return
        else:
            print("Please enter Yes or No (or 'exit' to skip).")

    user_comment = input("Please give us a reason for your answer. This will help us improve: ").strip()
    langfuse.score_current_trace(
        name="usefulness",
        value=feedback_value,
        data_type="BOOLEAN",
        comment=user_comment
    )