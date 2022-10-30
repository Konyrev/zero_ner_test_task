class QASpan:
    def __init__(self, token: str, label: str, start_context_char_pos: int, end_context_char_pos: int):
        self.token = token
        self.label = label
        self.start_context_char_pos = start_context_char_pos
        self.end_context_char_pos = end_context_char_pos


class QAInstance:
    def __init__(self, context, question, answer):
        self.context = context
        self.question = question
        self.answer = answer