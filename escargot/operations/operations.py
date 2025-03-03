from __future__ import annotations
import logging
from enum import Enum
from typing import List, Iterator, Dict, Callable, Union
from abc import ABC, abstractmethod
import itertools
import copy
import re
from .utils import remove_quotes, process_knowledge_ids, apply_function

from escargot.operations.thought import Thought
from escargot.language_models import AbstractLanguageModel
from escargot.prompter import ESCARGOTPrompter
from escargot.parser import ESCARGOTParser
from escargot.coder import Coder

class OperationType(Enum):
    """
    Enum to represent different operation types that can be used as unique identifiers.
    """

    generate: int = 0


class Operation(ABC):
    """
    Abstract base class that defines the interface for all operations.
    """

    _ids: Iterator[int] = itertools.count(0)

    operation_type: OperationType = None

    def __init__(self) -> None:
        """
        Initializes a new Operation instance with a unique id, and empty predecessors and successors.
        """
        self.logger: logging.Logger = None
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.executed: bool = False
        self.coder = None

    def can_be_executed(self) -> bool:
        """
        Checks if the operation can be executed based on its predecessors.

        :return: True if all predecessors have been executed, False otherwise.
        :rtype: bool
        """
        return all(predecessor.executed for predecessor in self.predecessors)

    def get_previous_thoughts(self) -> List[Thought]:
        """
        Iterates over all predecessors and aggregates their thoughts.

        :return: A list of all thoughts from the predecessors.
        :rtype: List[Thought]
        """
        previous_thoughts: List[Thought] = [
            thought
            for predecessor in self.predecessors
            for thought in predecessor.get_thoughts()
        ]

        return previous_thoughts

    def add_predecessor(self, operation: Operation) -> None:
        """
        Add a preceding operation and update the relationships.

        :param operation: The operation to be set as a predecessor.
        :type operation: Operation
        """
        self.predecessors.append(operation)
        operation.successors.append(self)

    def add_successor(self, operation: Operation) -> None:
        """
        Add a succeeding operation and update the relationships.

        :param operation: The operation to be set as a successor.
        :type operation: Operation
        """
        self.successors.append(operation)
        operation.predecessors.append(self)

    def execute(
        self, lm: AbstractLanguageModel, prompter: ESCARGOTPrompter, parser: ESCARGOTParser, got_steps: Dict, logger : logging.Logger, coder : Coder, **kwargs
    ) -> None:
        """
        Execute the operation, assuring that all predecessors have been executed.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: ESCARGOTPrompter
        :param parser: The parser for parsing responses.
        :type parser: ESCARGOTParser
        :param got_steps: The dictionary of steps in the Graph of Operations.
        :type got_steps: Dict
        :param kwargs: Additional parameters for execution.
        """
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger = logger
        self.coder = coder
        self.logger.debug(
            "Beginning execution of operation %d of type %s", self.id, self.operation_type
        )
        self._execute(lm, prompter, parser, got_steps, **kwargs)
        self.logger.debug("Operation %d executed", self.id)
        self.executed = True

    @abstractmethod
    def _execute(
        self, lm: AbstractLanguageModel, prompter: ESCARGOTPrompter, parser: ESCARGOTParser, got_steps: Dict, **kwargs
    ) -> None:
        """
        Abstract method for the actual execution of the operation.
        This should be implemented in derived classes.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: ESCARGOTPrompter
        :param parser: The parser for parsing responses.
        :type parser: ESCARGOTParser
        :param kwargs: Additional parameters for execution.
        """
        pass

    @abstractmethod
    def get_thoughts(self) -> List[Thought]:
        """
        Abstract method to retrieve the thoughts associated with the operation.
        This should be implemented in derived classes.

        :return: List of associated thoughts.
        :rtype: List[Thought]
        """
        pass


class Generate(Operation):
    """
    Operation to generate thoughts.
    """

    operation_type: OperationType = OperationType.generate

    def __init__(
        self, num_branches_prompt: int = 1, num_branches_response: int = 1
    ) -> None:
        """
        Initializes a new Generate operation.

        :param num_branches_prompt: Number of responses that each prompt should generate (passed to prompter). Defaults to 1.
        :type num_branches_prompt: int
        :param num_branches_response: Number of responses the LM should generate for each prompt. Defaults to 1.
        :type num_branches_response: int
        """
        super().__init__()
        self.num_branches_prompt: int = num_branches_prompt
        self.num_branches_response: int = num_branches_response
        self.thoughts: List[Thought] = []

    def get_thoughts(self) -> List[Thought]:
        """
        Returns the thoughts associated with the operation.

        :return: List of generated thoughts.
        :rtype: List[Thought]
        """
        return self.thoughts
    
    def generate_from_single_thought(
        self, lm: AbstractLanguageModel, prompter: ESCARGOTPrompter, parser: ESCARGOTParser, base_state: Dict
    ):
        prompts = []
        responses = []
        new_states = []
        if len(self.thoughts) > 0 and self.thoughts[-1].state["phase"] == "output":
            temp_state = copy.deepcopy(base_state)
            temp_state["phase"] = "output"
            temp_state['input'] = self.coder.step_output
            prompts.append(prompter.generate_prompt( **temp_state))
            self.logger.debug("Prompt for LM: \n%s", prompts[-1])
        elif "StepID" in base_state and "instruction" in base_state and base_state["instruction"]["Code"] is not None:
            code = base_state["instruction"]["Code"][0]
            new_code, compiled = self.coder.execute_code(code, base_state["instruction"]["Instruction"], base_state["StepID"], prompter, self.logger, base_state["full_code"])
            new_state  = {**base_state, "input": new_code, "compiled": compiled}
            new_state["instructions"][int(base_state["StepID"])-1]['Code'] = [new_code]
            new_states.append(new_state)
            self.logger.info("Code: %s", new_code)
            self.logger.info("Compiled: %s", compiled)
            
            return prompts, responses, new_states
        else:
            prompts.append(prompter.generate_prompt( **base_state))
        for prompt in prompts:
            if prompt is None or prompt == "":
                self.logger.warning("Prompt for LM is empty")
                return prompt, []
            self.logger.debug("Prompt for LM: %s", prompt)
            
            tries = 0
            while tries < 3:
                try:
                    new_states = []
                    responses = []
                    lm_responses =lm.get_response_texts(
                        lm.query(prompt, num_responses=self.num_branches_response)
                    )
                    for response in lm_responses:
                        responses.append(response)
                        if len(self.thoughts) > 0 and self.thoughts[-1].state["phase"] == "output":
                            new_state = parser.parse_generate_answer(temp_state, response)
                        else:
                            new_state = parser.parse_generate_answer(base_state, response)
                        new_states.append(new_state)
                        self.logger.debug("Response from LM: %s", response)
                    break
                except Exception as e:
                    self.logger.warning("Error in LM: %s, trying again with prompt: %s", e, prompt)
                    tries += 1

        if new_states != [] and "select_highest_score" in new_states[0]:
            #empty list of length len(new_states)
            scores = [new_state["scores"] for new_state in new_states]
            #element wise sum of the scores
            sum_scores = [sum(elem) for elem in zip(*scores)]
            #get the index of the highest score
            highest_score_index = sum_scores.index(max(sum_scores))
            responses = [sum_scores]
            new_states = [new_states[0]]
            new_states[0]["input"] = base_state["input"][highest_score_index]
            new_states[0]["scores"] = sum_scores
            if "full_plan" not in new_states[0] and (new_states[0]["phase"] == "python_conversion" or new_states[0]["phase"] == "plan_multihop"):
                new_states[0]["full_plan"] = new_states[0]["input"]
                self.logger.warning("Strategy:\n%s", new_states[0]["input"])
            new_states[0].pop("select_highest_score")
            
        
        return prompts, responses, new_states

    def _execute(
        self, lm: AbstractLanguageModel, prompter: ESCARGOTPrompter, parser: ESCARGOTParser, got_steps: Dict, **kwargs
    ) -> None:
        """
        Executes the Generate operation by generating thoughts from the predecessors.
        The thoughts are generated by prompting the LM with the predecessors' thought states.
        If there are no predecessors, the kwargs are used as a base state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: ESCARGOTPrompter
        :param parser: The parser for parsing responses.
        :type parser: ESCARGOTParser
        :param got_steps: The dictionary of steps in the Graph of Operations.
        :type got_steps: Dict
        :param kwargs: Additional parameters for execution.
        """
        previous_thoughts: List[Thought] = self.get_previous_thoughts()

        if len(previous_thoughts) == 0 and len(self.predecessors) > 0:
            return

        if len(previous_thoughts) == 0:
            # no predecessors, use kwargs as base state
            previous_thoughts = [Thought(state=kwargs)]

        if len(previous_thoughts) == 1:
        # for thought in previous_thoughts:
            base_state = previous_thoughts[0].state
            if "generate_successors" in base_state:
                base_state.pop("generate_successors")
            if "num_branches_response" in base_state:
                self.num_branches_response = base_state.pop("num_branches_response")
            
        elif len(previous_thoughts) > 1:

            base_states = [thought.state for thought in previous_thoughts]
            input_array = {}
            for base_state in base_states:
                input_array[base_state["StepID"]] = base_state["input"]
            base_state = copy.deepcopy(base_states[0])
            base_state["input"] = input_array
            if "instruction" in base_state:
                base_state.pop("instruction")
        if len(self.get_thoughts()) > 0 and "StepID" in self.get_thoughts()[0].state:
            base_state["StepID"] = self.get_thoughts()[0].state["StepID"]
            for instruction in base_state["instructions"]:
                if instruction["StepID"] == base_state["StepID"] or instruction["StepID"] == int(base_state["StepID"]) or instruction["StepID"] == str("StepID_" + base_state["StepID"]):
                    base_state["instruction"] = instruction
                    break
            
        prompts, responses, new_states = self.generate_from_single_thought(
            lm, prompter, parser, base_state
        )
        self.logger.info("Responses from LM: %s", responses)
        index = 0
        for new_state in new_states:
            if new_state is not None:
                if len(previous_thoughts) > 1 and len(prompts) >= 1:
                    new_state = {**base_state, **new_state, "prompt": prompts[index]}
                elif len(prompts) >= 1:
                    new_state = {**base_state, **new_state, "prompt": prompts[0]}
                else:
                    new_state = {**base_state, **new_state, "prompt": prompts}
                if len(self.thoughts) > 0:
                    if self.thoughts[-1].state["phase"] == "plan_assessment" or self.thoughts[-1].state["phase"] == "code_assessment":
                        if type(self.thoughts[-1].state["input"]) == str:
                            self.thoughts[-1].state["input"] = [self.thoughts[-1].state["input"]]
                        self.thoughts[-1].state["input"].append(new_state["input"])
                        if self.thoughts[-1].state["phase"] == "code_assessment":
                            self.thoughts[-1].state["full_code"] = new_state["input"]
                        continue
                    elif self.thoughts[-1].state["phase"] == "output":
                        self.thoughts[-1].state["input"] = new_state["input"]
                        self.thoughts[-1].state["prompt"] = new_state["prompt"]
                    else:
                        self.thoughts[-1].state = new_state
                else:
                    self.thoughts.append(Thought(new_state))

                if "generate_successors" in new_state:
                    predecessor = self
                    gen_nda = Generate(1, new_state["generate_successors"])
                    gen_nda.add_predecessor(predecessor)
                        
                
                # edges look like ['StepID_1-StepID_2', 'StepID_1-StepID_3', 'StepID_2-StepID_4', 'StepID_3-StepID_4']
                # it can also look like ['1-2', '1-3', '2-4', '3-4']
            
                if "edges" in new_state and "StepID" not in new_state:
                    #only one edge
                    if new_state["edges"] == []:
                        got_steps['1'] = Generate(1, 1)
                        got_steps['1'].thoughts = [Thought(
                            state={**self.thoughts[-1].state, "StepID": '1'}
                        )]
                    else:
                        for edge in new_state["edges"]:
                            edge = edge.split('-')
                            #remove "StepID" from each element in edge
                            edge = [elem.replace("StepID_", "") for elem in edge]
                            edge = [elem.replace("StepID", "") for elem in edge]

                            #sort the edge
                            edge.sort()

                            if edge[0] not in got_steps:
                                got_steps[edge[0]] = Generate(1, 1)
                                got_steps[edge[0]].thoughts = [Thought(
                                    state={**self.thoughts[-1].state, "StepID": edge[0]}
                                )]
                            if edge[1] not in got_steps:
                                got_steps[edge[1]] = Generate(1, 1)
                                got_steps[edge[1]].thoughts = [Thought(
                                    state={**self.thoughts[-1].state, "StepID": edge[1]}
                                )]
                            got_steps[edge[0]].add_successor(got_steps[edge[1]])
                        #check for all steps in got_steps for predecessors. If no predecessors, assign the self as predecessor
                    for step in got_steps:
                        if len(got_steps[step].predecessors) == 0:
                            got_steps[step].add_predecessor(self)
                    
                    #get the last integer key in the got_steps
                    got_steps_keys = list(got_steps.keys())
                    got_steps_keys = [int(elem) for elem in got_steps_keys]
                    got_steps_keys.sort()
                    got_steps_keys = [str(elem) for elem in got_steps_keys]
                    last_step = got_steps_keys[-1]
                    got_steps[str(last_step)].add_successor(Generate(1, 1))
                    got_steps[str(last_step)].successors[-1].thoughts = [Thought(
                        state={**self.thoughts[-1].state, "phase": "output"}
                    )]
                    got_steps[str(last_step)].successors[-1].thoughts[-1].state["input"] = ""
                    new_state.pop("edges")
                    
            index += 1

        if self.thoughts[-1].state["phase"] == "output":
            self.logger.info("Output: %s", self.thoughts[-1].state["input"])
        self.logger.debug(
            "Generate operation %d created %d new thoughts", self.id, len(self.thoughts)
        )

