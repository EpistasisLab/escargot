from __future__ import annotations
from typing import Dict, List
import re
import logging
import numpy as np
from escargot.prompter import ESCARGOTPrompter
import ast

def determine_and_execute(code_snippet, namespace={}):
    local_context = {}
    #detect if there is a print statement in the code
    if 'print' in code_snippet:
        code_snippet = code_snippet.replace('print','')
    try:
        # Try to parse as an expression
        ast.parse(code_snippet, mode='eval')
        # If successful, it's an expression
        return eval(code_snippet, namespace, local_context), 'eval', local_context
    except SyntaxError:
        # If there's a syntax error, try to parse as statements
        exec(code_snippet, namespace, local_context)
        return None, 'exec', local_context

class Coder:
    """
    Coder class to manage the code generation and execution.
    """

    def __init__(self) -> None:
        """
        Initialize the Coder instance with the logger.
        """
        self.local_context = {}
        self.step_output = {}

    def execute_code(self, code: str, instruction: str, step_id: str, prompter: ESCARGOTPrompter, logger: logging.Logger) -> str:
        """
        Execute the code and return the output.

        :param code: The code to be executed.
        :type code: str
        :return: The output of the code execution.
        :rtype: str
        """
        tries = 3
        compiled = False

        def knowledge_extract(request):
            return prompter.get_knowledge(request,instruction)
        # Add the knowledge_extract function to the local context
        self.local_context["knowledge_extract"] = knowledge_extract
        self.local_context["prompter"] = prompter

        backup_local_context = self.local_context.copy()
        
        while tries > 0 and not compiled:
            try:
                #detect long spaces within the code and remove them, but keep \n
                code = code.replace('            ','')
                result, expression_type, local_context = determine_and_execute(code, self.local_context)
                compiled = True
            except Exception as e:
                logger.warning(f"Could not execute code: {code}. Encountered exception: {e}")
                self.local_context = backup_local_context.copy()
                #debug using the prompter and using the error message
                prompt = prompter.generate_debug_code_prompt(code, instruction, e)
                code =prompter.lm.get_response_texts(
                    prompter.lm.query(prompt, num_responses=1)
                )[0]
                tries -= 1
        if compiled:
            #check for diff between local_context and backup_local_context
            if expression_type == 'eval':
                self.step_output[step_id] = result
            else:
                self.step_output[step_id] = local_context
                self.local_context = self.local_context | local_context
        logger.info(f"Step output: {self.step_output}")

        return code,compiled
    