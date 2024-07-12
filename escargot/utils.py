
import json

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
