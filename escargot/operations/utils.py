def remove_quotes(s):
    if s and s[0] in ("'", '"') and s[0] == s[-1]:
        return s[1:-1]
    return s

def process_knowledge_ids(knowledge_ids):
    knowledge_ids = knowledge_ids[knowledge_ids.find("(")+1:knowledge_ids.find(")")]
    knowledge_ids = [remove_quotes(elem.strip()) for elem in knowledge_ids.split(",")]
    return knowledge_ids

def apply_function(knowledge_ids, func_name, knowledge_list):
    intersect, union, difference = set(), set(), set()
    
    def update_set(op_set, current_list, op):
        if op:
            if not op_set:
                op_set.update(current_list)
            else:
                op_set.update(current_list) if op == 'union' else op_set.difference_update(current_list)

    for knowledge_id in knowledge_ids:
        if isinstance(knowledge_id, list):
            current_set = set(knowledge_id)
        else:
            current_set = set(knowledge_list.get(knowledge_id, []))
        if func_name == 'intersect':
            update_set(intersect, current_set, 'intersect')
        elif func_name == 'union':
            update_set(union, current_set, 'union')
        elif func_name == 'difference':
            update_set(difference, current_set, 'difference')

    return list(intersect or union or difference)

def get_knowledge_list_from_input(new_state, logger):
    #new_state["input"] is a string and needs to be converted to an array. Need to do a try except to catch any errors
    try:
        if new_state["input"] == "[]":
            knowledge_list_array = []
        #check if the array elements has "" or '' around them. If so add them for each elemen
        elif new_state["input"].count("'") >= 2 or new_state["input"].count('"') >= 2:
            knowledge_list_array = new_state["input"].replace("[", "").replace("]", "").replace("\"","").replace("'", "").split(",")
        elif new_state["input"].count("'") == 0 and new_state["input"].count('"') == 0:
            knowledge_list_array = new_state["input"].replace("[", "").replace("]", "").split(",")
        else:
            knowledge_list_array = eval(new_state["input"])
    except Exception as e:
        logger.error(f"Could not convert input to array: {new_state['input']}. Encountered exception: {e}")
        knowledge_list_array = []
    knowledge_list_array = [elem.strip() for elem in knowledge_list_array]
    return knowledge_list_array