import re
import logging
import json
import xml.etree.ElementTree as ET
import logging

def strip_answer_helper(text: str, tag: str = "") -> str:
    """
    Helper function to remove tags from a text.

    :param text: The input text.
    :type text: str
    :param tag: The tag to be stripped. Defaults to "".
    :type tag: str
    :return: The stripped text.
    :rtype: str
    """

    text = text.strip()
    if "Output:" in text:
        text = text[text.index("Output:") + len("Output:") :].strip()
    if tag != "":
        start = text.rfind(f"<{tag}>")
        end = text.rfind(f"</{tag}>")
        if start != -1 and end != -1:
            text = text[start + len(f"<{tag}>") : end].strip()
        elif start != -1:
            logging.warning(
                f"Only found the start tag <{tag}> in answer: {text}. Returning everything after the tag."
            )
            text = text[start + len(f"<{tag}>") :].strip()
        elif end != -1:
            logging.warning(
                f"Only found the end tag </{tag}> in answer: {text}. Returning everything before the tag."
            )
            text = text[:end].strip()
        else:
            logging.warning(
                f"Could not find any tag {tag} in answer: {text}. Returning the full answer."
            )
    return text

#strip answer helper but returns all instances of the tag
def strip_answer_helper_all(text: str, tag: str = "") -> str:
    """
    Helper function to remove tags from a text.

    :param text: The input text.
    :type text: str
    :param tag: The tag to be stripped. Defaults to "".
    :type tag: str
    :return: The stripped text.
    :rtype: str
    """

    text = text.strip()
    #get all instances of the tag
    start = [m.start() for m in re.finditer(f"<{tag}>", text)]
    end = [m.start() for m in re.finditer(f"</{tag}>", text)]
    # print(start)
    # print(end)
    return [text[text.index(f"<{tag}>", start[i]) + len(f"<{tag}>") : end[i]].strip() for i in range(len(start))]

def parse_xml(xml_data):
    # Parse the XML string
    #find <?xml version="1.0" encoding="UTF-8"?> and remove it
    xml_data = re.sub(r"<\?xml version=\"1.0\" encoding=\"UTF-8\"\?>", "", xml_data)
    #find ```xml and remove it
    xml_data = re.sub(r"```xml", "", xml_data)
    #find ``` and remove it
    xml_data = re.sub(r"```", "", xml_data)
    #find <Root> and remove it
    xml_data = re.sub(r"<Root>", "", xml_data)
    #find </Root> and remove it
    xml_data = re.sub(r"</Root>", "", xml_data)
    try:
        xml_data = '<Root>' + xml_data + '</Root>'
        root = ET.fromstring(xml_data)
    except Exception as e:
        logging.error(f"Could not parse XML data: {xml_data}. Encountered exception: {e}")

    def get_step(step):
        step_id = step.find('StepID').text
        instruction = step.find('Instruction')

        if step is None:
            return None  # Handle cases where there's no instruction element

        # Initialize empty lists to store information
        knowledge_requests = []
        for_info = None
        function = []

        # Check for KnowledgeRequest elements (can be multiple)
        for knowledge_request in step.findall('KnowledgeRequest'):
            knowledge_id = knowledge_request.find('KnowledgeID').text if knowledge_request.find('KnowledgeID') is not None else None
            node = knowledge_request.find('Query').text if knowledge_request.find('Query') is not None else None
            knowledge_requests.append({
                "KnowledgeID": knowledge_id,
                "Query": node
            })

        # Check for For element
        if step.find('For') is not None:
            for_var = step.find('For/ForVariable').text
            for_node = step.find('For/ForFunction/KnowledgeRequest/Node').text
            for_info = {
                "ForVariable": for_var,
                "ForNode": for_node
            }

        # Check for Function element
        # if step.find('Function') is not None:
        #     function = step.find('Function').text.strip()

        for function_element in step.findall('Function'):
            function.append(function_element.text.strip())


        # Return a list with relevant information (adjust as needed)
        return {
            "StepID": step_id,
            "InstructionType": "KnowledgeRequest" if knowledge_requests else "For" if for_info else "Function" if function else None,
            "Instruction": instruction.text.strip() if instruction.text else "",
            "KnowledgeRequests": knowledge_requests if knowledge_requests else None,
            # "For": for_info,
            "Function": function if function else None
        }

    # print('xml_data:',xml_data.split("\n"))
    # Extract and print details for each step
    instructions = root.find('Instructions').findall('Step')
    steps = []
    for step in instructions:
        parsed_step = get_step(step)
        # print(parsed_step)
        steps.append(parsed_step)

    # Extract and print edges
    #check if EdgeList exists
    if root.find('EdgeList') is None:
        return steps, []
    edges = root.find('EdgeList').findall('Edge')
    #remove \n from the text and whitespace
    edges = [edge.text.replace("\n","").strip() for edge in edges]
    # for edge in edges:
    #     print(f'Edge: {edge.text}')
    # print("edges:", edges)
    return steps, edges

def output_controller(operations_graph):

    output = []
    # for operation in operations_log['MCQ_1hop.json']['Which of the following binds to the drug Leucovorin? 1. CAD 2. PDS5B 3. SEL1L 4. ABCC2 5. RMI1']:
    # print(len(operations_graph))
    index = 0
    operation = operations_graph[0]
    while len(operation.successors) > 0:
    # for operation in operations_graph:
        
        operation_serialized = {
            "id": "node_"+str(index),
            "operation": operation.operation_type.name,
            "thoughts": [thought.state for thought in operation.get_thoughts()],
        }
        print(operation_serialized["thoughts"][0]["prompt"])
        
        output.append(operation_serialized)
        index = index + 1
        operation = operation.successors[0]
    edge_data = []
    num_of_branches = (len(operations_graph)-3)
    for i in range(0, int(num_of_branches),2):
        # edge_data.append([0, i+1])
        edge_data.append(["node_0", "node_"+str(i+1)])
        # edge_data.append([i+1, i+2])
        edge_data.append(["node_"+str(i+1), "node_"+str(i+2)])
        # edge_data.append([i+2, len(operations_graph)-2])
        edge_data.append(["node_"+str(i+2), "node_"+str(len(operations_graph)-2)])

    # edge_data.append([len(operations_graph)-2, len(operations_graph)-1])
    edge_data.append(["node_"+str(len(operations_graph)-2), "node_"+str(len(operations_graph)-1)])
    print(json.dumps(output, indent=4))
    print(json.dumps(edge_data, indent=4))

def final_operation(operations_graph):
    output = []
    operation = operations_graph[0]
    while len(operation.successors) > 0:
        operation = operation.successors[0]
    print(operation)
    return operation
