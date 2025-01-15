import websocket
import json
import logging
import time
import sys
import os 
from utils import retrieve_dataset_columns, retrieve_file_descriptions, retrieve_plans, retrieve_from_chat_history
import dill
import shutil

class MultiAgentManager:
    def __init__(self, config, agents, chat_id = None, message_id = None, debug_level = 1):
        self.config = config
        self.message_id = 0

        # the logs are in the logs folder. the logs folder has subfolders for each chat. each of these subfolders is named with the chat_id
        # within the chat_id folder, there are log files for each message. the log files are named with the message_id
        # the logs folder is created if it does not exist
        if not os.path.exists('logs'):
            os.mkdir('logs')
        if chat_id is not None:
           self.chat_id = chat_id
        else:
            #chat_id is based on the number of folders in the logs folder
            folders = os.listdir('logs')
            self.chat_id = len(folders)
        if not os.path.exists('logs/'+str(self.chat_id)):
            os.mkdir('logs/'+str(self.chat_id))
            

        #based on how many pickle files are in the chat_id folder, set the message_id to the next number
        files = os.listdir('logs/'+str(self.chat_id))
        for file in files:
            if file.startswith('checkpoint-'):
                message_id = int(file.split('-')[1].split('.')[0])
                if message_id > self.message_id:
                    self.message_id = message_id

        #get history from ('logs/'+str(chat_id)+'/chat_history.json')
        if os.path.exists('logs/'+str(self.chat_id)+'/chat_history.json'):
            with open('logs/'+str(self.chat_id)+'/chat_history.json', 'r') as f:
                chat_history = json.load(f)
                
            chat_history = {int(k): v for k, v in chat_history.items()}
            self.chat_history = chat_history
        else:
            self.chat_history = {}

        self.agents = agents
        self.debug_level = debug_level

        print("MultiAgentManager initialized with the following agents:")
        for agent in self.agents:
            print(agent)
        print(f"Chat ID: {self.chat_id}")
        print(f"Message ID: {self.message_id}")
        print(f"Current chat history:")
        print(json.dumps(self.chat_history, indent=2))

        

    def direct_chat(self, message, answer_type = "natural", message_id = None):
        if message_id is not None:
            self.message_id = message_id
        else:
            self.message_id += 1
        print("Running escargot...")
        if message.startswith('@'):
            #the message will be in the format "@agent_name message"
            #split the message by space
            message = message.split(' ')
            #get the agent name
            escargot_agent = message[0][1:]
            #get the content
            content = ' '.join(message[1:])

            if escargot_agent in self.agents:
                agent = self.agents[escargot_agent]['agent']
                print(f"Running {escargot_agent}...")
                files = os.listdir('logs/'+str(self.chat_id))
                if hasattr(agent, "file_descriptions"):
                    file_descriptions, plans = retrieve_from_chat_history(self.chat_history, self.chat_id, self.message_id)
                    agent.file_descriptions = file_descriptions
                    agent.plans = plans
                    plans += "\n" + content
                if hasattr(agent, "dataframe_columns"):
                    agent.dataframe_columns = retrieve_dataset_columns()

                os.chdir('logs/'+str(self.chat_id))
                result = agent.ask(content, debug_level=self.debug_level, answer_type=answer_type)
                os.chdir('../..')
                if agent.controller.final_thought.state['phase'] == 'output':
                    if answer_type == 'array' and len(result) > 0:
                        #access the result dictionary keys and stringify it
                        keys = list(result.keys())
                        result = result[keys[0]]
                    new_files = list(set(os.listdir('logs/'+str(self.chat_id))) - set(files))
                    with open('logs/'+str(self.chat_id)+'/checkpoint-'+str(self.message_id)+'.pkl', 'wb') as f:
                        dill.dump(result, f)

                    for file in new_files:
                        if file.endswith('.png') or file.endswith('.jpg') or file.endswith('.jpeg'):
                            #display the image
                            print(f"Displaying image: {file}")
                            #display the images
                            from PIL import Image
                            img = Image.open('logs/'+str(self.chat_id)+'/'+file)
                            img.show()
                            

                        elif file.endswith('.csv'):
                            #display the csv
                            print(f"Displaying csv: {file}")
                            #show the head of the csv
                            with open('logs/'+str(self.chat_id)+'/'+file, 'r') as f:
                                print(f.read())

                        elif file.endswith('.pkl'):
                            #preview what's in the pickle file
                            print(f"Displaying pickle file: {file}")
                            with open('logs/'+str(self.chat_id)+'/'+file, 'rb') as f:
                                print(dill.load(f))
            

                else:
                    print(f'{escargot_agent} error')
                    return None

                
                self.chat_history[self.message_id] = {'agent': escargot_agent, 'message': content, 'result': result}
                with open('logs/'+str(self.chat_id)+'/chat_history.json', 'w') as f:
                    json.dump(self.chat_history, f)
                return result
        else:
            print("Error: message must start with '@'")
        return None 
            
    #function to save the current chat to a new one
    #copies over all assets
    def save_chat(self, chat_id):
        if not os.path.exists('logs/'+str(chat_id)):
            os.mkdir('logs/'+str(chat_id))
        files = os.listdir('logs/'+str(self.chat_id))
        #copy over all files with shutil
        for file in files:
            #make sure it's not a directory
            if os.path.isfile('logs/'+str(self.chat_id)+'/'+file):
                shutil.copy('logs/'+str(self.chat_id)+'/'+file, 'logs/'+str(chat_id)+'/'+file)
        print(f"Chat {self.chat_id} saved to chat {chat_id}")
        self.chat_id = chat_id
