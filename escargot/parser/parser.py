from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Union
import logging
from .utils import strip_answer_helper, strip_answer_helper_all, parse_xml


class Parser(ABC):
    """
    Abstract base class that defines the interface for all parsers.
    Parsers are used to parse the responses from the language models.
    """

    @abstractmethod
    def parse_generate_answer(self, state: Dict, texts: List[str]) -> List[Dict]:
        """
        Parse the response from the language model for a generate prompt.

        :param state: The thought state used to generate the prompt.
        :type state: Dict
        :param texts: The responses to the prompt from the language model.
        :type texts: List[str]
        :return: The new thought states after parsing the response from the language model.
        :rtype: List[Dict]
        """
        pass


class ESCARGOTParser(Parser):
    """
    ALZKBParser provides the parsing of language model reponses specific to the
    ALZKB example.

    Inherits from the Parser class and implements its abstract methods.
    """

    def __init__(self) -> None:
        """
        Inits the response cache.
        """
        self.cache = {}

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
            if state["method"] == "got":
                try:
                    if state["phase"] == "planning":
                        new_state = state.copy()
                        new_state["input"] = text
                        #skipping plan_assessment phase for now
                        # new_state["phase"] = "plan_assessment"
                        new_state["phase"] = "plan_assessment"
                        new_state["generate_successors"] = 1
                        # print("planning:", text)
                    elif state["phase"] == "plan_assessment":
                        new_state = state.copy()
                        #convert text to json and select from the input the top strategy
                        try:
                            text = strip_answer_helper_all(text, "Approach")
                            # get the highest score and the approach
                            approach = max(text, key=lambda x: int(strip_answer_helper(x,"Score")))
                            approach = strip_answer_helper(approach, "ApproachID")
                            approach = int(approach)-1
                            new_state["input"] = new_state["input"][approach]
                        except Exception as e:
                            logging.error(f"Could not convert text to xml: {text}. Encountered exception: {e}")

                        new_state["phase"] = "xml_conversion"
                        new_state["generate_successors"] = 1
                    elif state["phase"] == "xml_conversion":
                        new_state = state.copy()
                        new_state["input"] = text
                        # new_state["phase"] = "xml_cleanup"
                        new_state["phase"] = "steps"
                        instructions, edges = parse_xml(text)
                        new_state["instructions"] = instructions
                        new_state["edges"] = edges
                    elif state["phase"] == "xml_cleanup":
                        new_state = state.copy()
                        new_state["input"] = text
                        new_state["phase"] = "steps"
                        instructions, edges = parse_xml(text)
                        new_state["instructions"] = instructions
                        new_state["edges"] = edges
                        print("instructions:", instructions)
                        print("edges:", edges)
                    elif state["phase"] == "steps":
                        new_state = state.copy()
                        new_state["input"] = text
                    elif state["phase"] == "output":
                        new_state = state.copy()
                        new_state["input"] = text
                except Exception as e:
                    logging.error(
                        f"Could not parse step answer: {text}. Encountered exception: {e}"
                    )
            
        return new_state
