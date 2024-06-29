from __future__ import annotations
import logging
from enum import Enum
from typing import List, Iterator, Dict, Callable, Union
from abc import ABC, abstractmethod
import itertools
import copy


from escargot.operations.thought import Thought
from escargot.language_models import AbstractLanguageModel
from escargot.prompter import Prompter
from escargot.parser import Parser



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
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)
        self.id: int = next(Operation._ids)
        self.predecessors: List[Operation] = []
        self.successors: List[Operation] = []
        self.executed: bool = False

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
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, got_steps: Dict, knowledge_list : Dict, **kwargs
    ) -> None:
        """
        Execute the operation, assuring that all predecessors have been executed.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
        :param kwargs: Additional parameters for execution.
        :raises AssertionError: If not all predecessors have been executed.
        """
        assert self.can_be_executed(), "Not all predecessors have been executed"
        self.logger.info(
            "Executing operation %d of type %s", self.id, self.operation_type
        )
        self._execute(lm, prompter, parser, got_steps, knowledge_list, **kwargs)
        self.logger.debug("Operation %d executed", self.id)
        self.executed = True

    @abstractmethod
    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, got_steps: Dict, knowledge_list : Dict, **kwargs
    ) -> None:
        """
        Abstract method for the actual execution of the operation.
        This should be implemented in derived classes.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
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
        self, lm: AbstractLanguageModel, prompter: Prompter, base_state: Dict, knowledge_list: Dict
    ):
        prompts = []
        responses = []
        #within a step, if there are knowledge requests needed, and there are multiple, the requests should be made separate
        if "StepID" in base_state:
            if "instruction" in base_state:
                if base_state["instruction"]["KnowledgeRequests"] is not None:
                    for knowledge_request in base_state["instruction"]["KnowledgeRequests"]:
                        temp_state = copy.deepcopy(base_state)
                        temp_state["instruction"]["KnowledgeRequests"] = [knowledge_request]
                        prompts.append(prompter.generate_prompt(self.num_branches_prompt, knowledge_list, **temp_state))
                
        else:
            prompts.append(prompter.generate_prompt(self.num_branches_prompt, knowledge_list, **base_state))
        for prompt in prompts:
            if prompt is None or prompt == "":
                self.logger.debug("Prompt for LM is empty")
                return prompt, []
            self.logger.debug("Prompt for LM: %s", prompt)
            if "StepID" in base_state and "instruction" in base_state and base_state["instruction"]["Function"] is not None:
                responses.append("Function: " + base_state["instruction"]["Function"])
            else:
                responses.append(lm.get_response_texts(
                    lm.query(prompt, num_responses=self.num_branches_response)
                ))
        return prompts, responses
    
    def generate_from_multiple_thoughts(
        self, lm: AbstractLanguageModel, prompter: Prompter, base_state: List[Dict], knowledge_list: Dict
    ):
        
        prompt = prompter.generate_prompt(self.num_branches_prompt, knowledge_list, **base_state)
        if prompt is None or prompt == "":
            self.logger.debug("Prompt for LM is empty")
            return prompt, []
        self.logger.debug("Prompt for LM: %s", prompt)
        if "StepID" in base_state and "instruction" in base_state and base_state["instruction"]["Function"] is not None:
            responses = ["Function: " + base_state["instruction"]["Function"]]
        else:
            responses = lm.get_response_texts(
                lm.query(prompt, num_responses=self.num_branches_response)
            )
        return prompt, responses

    def _execute(
        self, lm: AbstractLanguageModel, prompter: Prompter, parser: Parser, got_steps: Dict, knowledge_list : Dict, **kwargs
    ) -> None:
        """
        Executes the Generate operation by generating thoughts from the predecessors.
        The thoughts are generated by prompting the LM with the predecessors' thought states.
        If there are no predecessors, the kwargs are used as a base state.

        :param lm: The language model to be used.
        :type lm: AbstractLanguageModel
        :param prompter: The prompter for crafting prompts.
        :type prompter: Prompter
        :param parser: The parser for parsing responses.
        :type parser: Parser
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
            if len(self.get_thoughts()) > 0 and "StepID" in self.get_thoughts()[0].state:
                base_state["StepID"] = self.get_thoughts()[0].state["StepID"]
                for instruction in base_state["instructions"]:
                    if instruction["StepID"] == base_state["StepID"] or instruction["StepID"] == int(base_state["StepID"]) or instruction["StepID"] == str("StepID_" + base_state["StepID"]):
                        base_state["instruction"] = instruction
                        break
                
            prompts, responses = self.generate_from_single_thought(
                lm, prompter, base_state, knowledge_list
            )
            
            self.logger.debug("Responses from LM: %s", responses)
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
            
            prompts, responses = self.generate_from_multiple_thoughts(
                lm, prompter, base_state, knowledge_list
            )
            self.logger.debug("Responses from LM: %s", responses)
        index = 0
        new_state = None
        for response in responses:
            new_state = parser.parse_generate_answer(base_state, response)
            if new_state is not None:
                new_state = {**base_state, **new_state, "prompt": prompts[index]}
                if len(self.thoughts) > 0:
                    self.thoughts[-1].state = new_state
                else:
                    self.thoughts.append(Thought(new_state))
                self.logger.debug(
                    "New thought %d created with state %s",
                    self.thoughts[-1].id,
                    self.thoughts[-1].state,
                )
                if "generate_successors" in new_state:
                    predecessor = self
                    for i in range(new_state["generate_successors"]):
                        gen_nda = Generate(1, 1)
                        gen_nda.add_predecessor(predecessor)
                
                # edges look like ['StepID_1-StepID_2', 'StepID_1-StepID_3', 'StepID_2-StepID_4', 'StepID_3-StepID_4']
                # it can also look like ['1-2', '1-3', '2-4', '3-4']
            
                if "edges" in new_state and "StepID" not in new_state:
                    # self.logger.info("Edges: %s", new_state["edges"])
                    for edge in new_state["edges"]:
                        edge = edge.split('-')
                        #remove "StepID" from each element in edge
                        edge = [elem.replace("StepID_", "") for elem in edge]
                        edge = [elem.replace("StepID", "") for elem in edge]

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
                    #get the last step in the got_steps and add a successor to it
                    last_step = list(got_steps.keys())[-1]
                    got_steps[last_step].add_successor(Generate(1, 1))
                    got_steps[last_step].successors[-1].thoughts = [Thought(
                        state={**self.thoughts[-1].state, "phase": "output"}
                    )]
                    got_steps[last_step].successors[-1].thoughts[-1].state["input"] = ""
                    self.logger.info("GoT Steps From XML: %s", got_steps)
                    
                #populate knowledge_list with condensed knowledge
                if new_state["phase"] == "steps" and "instruction" in new_state:
                    if new_state["instruction"]["KnowledgeRequests"] is not None:
                        #try eval() to convert string to list
                        #new_state["input"] is a string and needs to be converted to an array. Need to do a try except to catch any errors
                        try:
                            if new_state["input"] == "[]":
                                knowledge_list_array = []
                            #check if the array elements has "" or '' around them. If so add them for each elemen
                            elif new_state["input"].count("'") >= 2 or new_state["input"].count('"') >= 2:
                                knowledge_list_array = new_state["input"].replace("[", "").replace("]", "").replace("'", "").split(",")
                            elif new_state["input"].count("'") == 0 and new_state["input"].count('"') == 0:
                                knowledge_list_array = new_state["input"].replace("[", "").replace("]", "").split(",")
                            else:
                                knowledge_list_array = eval(new_state["input"])
                        except:
                            # print("Error in converting knowledge string to eval:", new_state["input"])
                            knowledge_list_array = []
                        knowledge_request = new_state["instruction"]["KnowledgeRequests"][index]
                        if knowledge_request["KnowledgeID"] not in knowledge_list:
                            #remove whitespaces from the knowledge request
                            knowledge_list_array = [elem.strip() for elem in knowledge_list_array]
                            knowledge_list[knowledge_request["KnowledgeID"]] = knowledge_list_array
                            # print("knowledge_ID:", knowledge_request["KnowledgeID"], knowledge_request["Node"])
                            # print("knowledge_list:", knowledge_list)
            index += 1

        #populate the 
        if new_state["phase"] == "steps" and new_state is not None and "instruction" in new_state and new_state["instruction"]["Function"] is not None:
            #parse the function and get the two knowledge lists to intersect
            # example: INTERSECT(Knowledge_1, Knowledge_2)
            # example: INTERSECT(Knowledge_3, Knowledge_6, Knowledge_7)
            # example: Intesect(Knowledge_1, Knowledge_2, Knowledge_3)
            
            #get the two knowledge lists by obtaining the first set of parantheses
            knowledge_ids = new_state["instruction"]["Function"]
            knowledge_ids = knowledge_ids[knowledge_ids.find("(")+1:knowledge_ids.find(")")]
            knowledge_ids = knowledge_ids.split(",")
            #remove any spaces in the list
            knowledge_ids = [elem.strip() for elem in knowledge_ids]
            if "intersect" in new_state["instruction"]["Function"].lower():
                #intersect the nonzero amount of elements in the list
                intersected_list = []
                for knowledge_id in knowledge_ids:
                    if knowledge_id in knowledge_list:
                        if len(intersected_list) == 0:
                            intersected_list = knowledge_list[knowledge_id]
                        else:
                            intersected_list = list(set(intersected_list) & set(knowledge_list[knowledge_id]))
                # print("intersected_list:", intersected_list)
                self.logger.info("intersected_list: %s", intersected_list)
                self.thoughts[-1].state["output"] = intersected_list
                knowledge_list["StepID_"+new_state["StepID"]] = self.thoughts[-1].state["output"]
            elif "union" in new_state["instruction"]["Function"].lower():
                #union the nonzero amount of elements in the list
                union_list = []
                for knowledge_id in knowledge_ids:
                    if knowledge_id in knowledge_list:
                        if len(union_list) == 0:
                            union_list = knowledge_list[knowledge_id]
                        else:
                            union_list = list(set(union_list) | set(knowledge_list[knowledge_id]))
                # print("union_list:", union_list)
                self.logger.info("union_list: %s", union_list)  
                self.thoughts[-1].state["output"] = union_list
                knowledge_list["StepID_"+new_state["StepID"]] = self.thoughts[-1].state["output"]
            elif "difference" in new_state["instruction"]["Function"].lower():
                #difference the nonzero amount of elements in the list
                difference_list = []
                for knowledge_id in knowledge_ids:
                    if knowledge_id in knowledge_list:
                        if len(difference_list) == 0:
                            difference_list = knowledge_list[knowledge_id]
                        else:
                            difference_list = list(set(difference_list) - set(knowledge_list[knowledge_id]))
                # print("difference_list:", difference_list)
                self.logger.info("difference_list: %s", difference_list)
                self.thoughts[-1].state["output"] = difference_list
                knowledge_list["StepID_"+new_state["StepID"]] = self.thoughts[-1].state["output"]
            else:
                self.logger.warning("Instruction function not recognized", new_state["instruction"]["Function"])
                # print("Instruction function not recognized", new_state["instruction"]["Function"])
            
            
        if (
            len(self.thoughts)
            > self.num_branches_prompt
            * self.num_branches_response
            * len(previous_thoughts)
            and self.num_branches_prompt > 0
        ):
            self.logger.warning(
                "Generate operation %d created more thoughts than expected",
                self.id,
            )
        self.logger.info(
            "Generate operation %d created %d new thoughts", self.id, len(self.thoughts)
        )

