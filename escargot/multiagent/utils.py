import re
import logging
import json
import xml.etree.ElementTree as ET
import logging
import os
import pandas as pd
import numpy as np

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

def parse_xml(xml_data, logger):
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
    xml_data = xml_data.replace("&", "&amp;")
    try:
        xml_data = '<Root>' + xml_data + '</Root>'
        root = ET.fromstring(xml_data)
    except Exception as e:
        logger.error(f"Could not parse XML data: {xml_data}. Encountered exception: {e}")
        return None, None

    def get_step(step):
        step_id = step.find('StepID').text
        instruction = step.find('Instruction')

        if step is None:
            return None  # Handle cases where there's no instruction element
        
        # Initialize empty lists to store information
        # codes = []
        # for info in step.findall('Code'):
        #     code_text = info.text.strip()
        #     if code_text:
        #         #unescape the code
        #         code_text = code_text.replace("&amp;", "&")
        #     codes.append(code_text)

        agent = step.find('Agent').text.strip() if step.find('Agent').text else ""

        # Return a list with relevant information (adjust as needed)
        return {
            "StepID": step_id,
            "Instruction": instruction.text.strip() if instruction.text else "",
            "Agent" : agent
            # "Code": codes
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


def parse_xml_code(xml_data, logger):
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
    xml_data = xml_data.replace("&", "&amp;")
    try:
        xml_data = '<Root>' + xml_data + '</Root>'
        root = ET.fromstring(xml_data)
    except Exception as e:
        logger.error(f"Could not parse XML data: {xml_data}. Encountered exception: {e}")
        return None, None

    def get_step(step):
        step_id = step.find('StepID').text
        instruction = step.find('Instruction')

        if step is None:
            return None  # Handle cases where there's no instruction element
        
        # Initialize empty lists to store information
        codes = []
        for info in step.findall('Code'):
            code_text = info.text.strip()
            if code_text:
                #unescape the code
                code_text = code_text.replace("&amp;", "&")
            codes.append(code_text)

        # agent = step.find('Agent').text.strip() if step.find('Agent').text else ""

        # Return a list with relevant information (adjust as needed)
        return {
            "StepID": step_id,
            "Instruction": instruction.text.strip() if instruction.text else "",
            # "Agent" : agent
            "Code": codes
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

def retrieve_file_descriptions(user_message_id, step_limit = float('inf')):
    #get all pkl files in the directory
    import dill as pickle
    pkl_files = [f for f in os.listdir() if f.endswith('.pkl') and f.startswith(str(user_message_id)+'-')]
    file_descriptions = []
    #load all pkl files
    for f in pkl_files:
        with open(f, 'rb') as file:
            obj = pickle.load(file)
            #split the file name and the extension
            step, ext = f.split('.')
            #split the step and the file name
            message_id, step_id = step.split('-')
            #if the step is greater than the limit, skip the file
            if int(step_id) > step_limit:
                continue

            pickle_info = {'file': f, 'Step': step_id}
            #describe the obj like its type and if needed its shape and keys.
            pickle_info['type'] = type(obj)
            if hasattr(obj, 'shape'):
                pickle_info['shape'] = obj.shape
            if hasattr(obj, 'keys'):
                pickle_info['keys'] = obj.keys()
                pickle_info['examples'] = str(obj)[:100]+'...'
                # for key in obj.keys():
                #     #stringity the values and put the first 100 characters in the description
                #     pickle_info['examples'][key] = str(obj[key])[:100]+'...'
            if hasattr(obj, 'describe'):
                pickle_info['describe'] = obj.describe()
            if hasattr(obj, 'info'):
                pickle_info['info'] = obj.info()
            if type(obj) == list:
                pickle_info['length'] = len(obj)
                pickle_info['peek inside'] = str(obj)[:100]+'...'
            if type(obj) == str:
                pickle_info['length'] = len(obj)
                pickle_info['peek inside'] = obj
            file_descriptions.append(pickle_info)

    return file_descriptions

def retrieve_plans(user_message_id, step_limit=float('inf')):
    #get the plans text file
    import logging 
    plans = [f for f in os.listdir() if f.endswith('.txt') and f.startswith(str(user_message_id)+'-')]
    if not plans:
        return []
    #read the plans file
    with open(plans[0], 'r') as file:
        plans = file.read()
    #parse the xml
    plans = parse_xml(plans, logger=logging)[0]

    plan_description = []
    for plan in plans:
        step_id = int(plan['StepID'])
        if step_id > step_limit:
            continue
        plan_description.append({'Step': plan['StepID'], 'Instruction': plan['Instruction']})
    return plan_description

def retrieve_dataset_columns():
    if not os.path.exists('../dataset.feather'):
        return []
    #read the dataset
    df = pd.read_feather('../dataset.feather')
    #return the columns
    return df.columns

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

def retrieve_from_chat_history(chat_history):
    #get all unique user_message_ids
    user_message_ids = set([message['user_message_id'] for message in chat_history if 'user_message_id' in message])

    all_file_descriptions = []
    all_plans = []
    
    for user_message_id in user_message_ids:

        #get the file descriptions
        file_descriptions = retrieve_file_descriptions(user_message_id)
        #concatenate the file descriptions
        all_file_descriptions += file_descriptions

    #get the plans
    for message in chat_history:
        if message['id'] in user_message_ids or message['message'].startswith('```escargot|SHOW```'):
            all_plans.append(message['message'])

    return all_file_descriptions, all_plans