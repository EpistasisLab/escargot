from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import logging
from .utils import strip_answer_helper, strip_answer_helper_all, parse_xml


class ESCARGOTParser:
    """
    ALZKBParser provides the parsing of language model reponses specific to the
    ALZKB example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self, logger: logging.Logger = None) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}
        self.logger = logger

    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        In GOT, the generate prompt is used for planning, plan assessment, xml conversion, knowledge extraction and array function 

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the respones from the language model.
        :rtype: List[Dict]
        """

        if type(texts) == str:
            texts = [texts]
        for text in texts:
            self.logger.debug(f"Got response: {text}")
            if state["method"] == "got":
                try:
                    if state["phase"] == "planning":
                        new_state = state.copy()
                        new_state["input"] = text
                        new_state["previous_phase"] = "planning"
                        new_state["phase"] = "plan_assessment"
                        new_state["num_branches_response"] = 3
                        new_state["generate_successors"] = 1
                    elif state["phase"] == "plan_assessment":
                        new_state = state.copy()
                        #convert text to json and select from the input the top strategy
                        scores = []
                        try:
                            text = strip_answer_helper_all(text, "Approach")
                            #get all the scores
                            for i in text:
                                scores.append(int(strip_answer_helper(i,"Score")))
                            new_state["scores"] = scores
                        except Exception as e:
                            self.logger.warning(f"Could not convert text to xml: {text}. Encountered exception: {e}")
                        # new_state["phase"] = "xml_conversion"
                        new_state["previous_phase"] = "plan_assessment"
                        new_state["phase"] = "python_conversion"
                        new_state["num_branches_response"] = 3
                        new_state["select_highest_score"] = 1
                        new_state["generate_successors"] = 1
                    elif state["phase"] == "python_conversion":
                        new_state = state.copy()
                        new_state.pop("scores")
                        new_state["input"] = text
                        self.logger.debug(f"Got Python code: {text}")
                        new_state["previous_phase"] = "python_conversion"
                        new_state["phase"] = "code_assessment"
                        new_state["generate_successors"] = 1
                    elif state["phase"] == "code_assessment":
                        new_state = state.copy()
                        scores = []
                        try:
                            text = strip_answer_helper_all(text, "Code")
                            for i in text:
                                scores.append(int(strip_answer_helper(i,"Score")))
                            new_state["scores"] = scores
                            # get the highest score and the approach
                            # approach = max(text, key=lambda x: int(strip_answer_helper(x,"Score")))
                            # approach = strip_answer_helper(approach, "CodeID")
                            # approach = int(approach)-1
                            # new_state["input"] = new_state["input"][approach]
                        except Exception as e:
                            self.logger.warning(f"Could not convert text to xml: {text}. Encountered exception: {e}")
                        new_state["previous_phase"] = "code_assessment"
                        new_state["phase"] = "xml_conversion"
                        new_state["select_highest_score"] = 1
                        new_state["generate_successors"] = 1
                        
                    elif state["phase"] == "xml_conversion":
                        new_state = state.copy()
                        new_state.pop("scores")
                        new_state["input"] = text
                        # new_state["phase"] = "xml_cleanup"
                        # new_state["generate_successors"] = 1
                        new_state["previous_phase"] = "xml_conversion"
                        new_state["phase"] = "steps"
                        instructions, edges = parse_xml(text, self.logger)
                        new_state["instructions"] = instructions
                        new_state["edges"] = edges
                        self.logger.info("Got instructions: \n%s",instructions)
                        self.logger.info("Got edges: \n%s",edges)
                    elif state["phase"] == "xml_cleanup":
                        new_state = state.copy()
                        new_state["input"] = text
                        new_state["previous_phase"] = "xml_cleanup"
                        new_state["phase"] = "steps"
                        instructions, edges = parse_xml(text, self.logger)
                        new_state["instructions"] = instructions
                        new_state["edges"] = edges
                        self.logger.info(f"Got instructions: {instructions}")
                        self.logger.info(f"Got edges: {edges}")
                    elif state["phase"] == "steps":
                        new_state["previous_phase"] = "steps"
                        new_state = state.copy()
                        new_state["input"] = text
                    elif state["phase"] == "output":
                        new_state = state.copy()
                        new_state["input"] = text
                except Exception as e:
                    self.logger.error(
                        f"Could not parse step answer: {text}. Encountered exception: {e}"
                    )
            
        return new_state
