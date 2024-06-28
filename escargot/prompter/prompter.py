# Copyright (c) 2023 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main authors: Robert Gerstenberger, Nils Blach

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List


class Prompter(ABC):
    """
    Abstract base class that defines the interface for all prompters.
    Prompters are used to generate the prompts for the language models.
    """

    @abstractmethod
    def generate_prompt(self, num_branches: int, knowledge_list:Dict, **kwargs) -> str:
        """
        Generate a generate prompt for the language model.
        The thought state is unpacked to allow for additional keyword arguments
        and concrete implementations to specify required arguments explicitly.

        :param num_branches: The number of responses the prompt should ask the LM to generate.
        :type num_branches: int
        :param kwargs: Additional keyword arguments.
        :return: The generate prompt.
        :rtype: str
        """
        pass
