import argparse
from prompt_config import *
import pandas as pd
global data_store 
data_store = {
	'question': [],
	'context': [],
	'answer': [],
	'ground_truth': [],
	'time': [],
	'source': [],
	}

# Initialize variables based on model selection
def inizialize(args):
	global intro_marker
	global context_marker
	global end_context_marker
	global answer_marker
	global model_id

	if args.model_name[0]=='Q':
		intro_marker=intro_marker_qwen
		context_marker=context_marker_qwen
		end_context_marker=end_context_marker_qwen
		answer_marker=answer_marker_qwen
		model_id=model_path+"Qwen/"+args.model_name

	if args.model_name[0]=='P':
		intro_marker=intro_marker_phi
		context_marker=context_marker_phi
		end_context_marker=end_context_marker_phi
		answer_marker=answer_marker_phi
		model_id=model_path+"microsoft/"+args.model_name

	return model_id

# Parse command-line arguments
def parse_args():
	parser = argparse.ArgumentParser(description="Run the question answering pipeline.")
	parser.add_argument('--model_name', type=str, required=True, help="The name or path of the model to be used.")
	parser.add_argument('--test', type=str, required=False, help="The name or path of the model to be used.")
	return parser.parse_args()


# Process and store question, context, and answer data
def process_question(model, question, answer, context, time, data):
	q=data['Question'][question]
	g=data['ground_truths'][question]

	# Append the new data to the respective lists in the dictionary
	data_store['question'].append(q)
	data_store['context'].append(context)
	data_store['answer'].append(answer)
	data_store['ground_truth'].append(g)
	data_store['time'].append(time)

	# Increment the count
	data_store['count'] += 1
	# Check if last questions have been processed
	if data_store['count'] == (len(data)):
		# Create a DataFrame from the stored data
		df = pd.DataFrame({
		'question': data_store['question'],
		'context': data_store['context'],
		'answer': data_store['answer'],
		'ground_truth': data_store['ground_truth'],
		'time': data_store['time']
		})

		# Save the DataFrame to a CSV file
		csv_filename = f"CSV_Result_med/{db}{model}_{file_name}"
		df.to_csv(csv_filename, index=False)

		# Print success message
		print(f"Data saved as '{csv_filename}'.")

		# Reset the data store for the next set of 50 questions
		data_store['question'].clear()
		data_store['answer'].clear()
		data_store['context'].clear()
		data_store['time'].clear()
		data_store['count'] = 0


# Split input string into context and answer based on defined prompt
def split_string(input_string):
	# Find the positions of the markers
	intro_pos = input_string.find(intro_marker)
	end_context_pos = input_string.find(end_context_marker)
	context_pos = input_string.find(context_marker)
	answer_pos = input_string.find(answer_marker)

	# Extract the answer (everything after answer_marker)
	context_ = input_string[context_pos + len(context_marker):end_context_pos].strip() if answer_pos != -1 else ""

	# Extract the answer (everything after answer_marker)
	answer_ = input_string[answer_pos + len(answer_marker):].strip() if answer_pos != -1 else ""

	return context_, answer_
 
 
 
 
# Split input string into context and answer based on defined prompt
def split_answer(input_string):
	# Find the positions of the markers
	answer_pos = input_string.find(answer_marker)
	
	# Extract the answer (everything after answer_marker)
	answer_ = input_string[answer_pos + len(answer_marker):].strip() if answer_pos != -1 else ""
	print(answer_)
	return answer_ 


def store_answer(db, model, file_name, llm_response, time, ground_truth_data, save_file):
	data_store['question'].append(llm_response['question'])
	data_store['answer'].append(split_answer(llm_response['answer']))
	data_store['ground_truth'].append(ground_truth_data)
	data_store['time'].append(time)
	
	temp_context=[]
	temp_source=[]
	context_=[]
	for n in range(0, len(llm_response['context'])):
		temp_source.append(llm_response['context'][n].metadata)
		context_.append(llm_response['context'][n].page_content)
		temp_context.append(context_)
		context_ = []
	
	# Append the new data to the respective lists in the dictionary
	data_store['context'].append(temp_context)
	data_store['source'].append(temp_source)
	
	if save_file:#(len(data)):
	# Create a DataFrame from the stored data
		df = pd.DataFrame({
		    'question': data_store['question'],
		    'context': data_store['context'],
		    'answer': data_store['answer'],
		    'ground_truth': data_store['ground_truth'],
		    'time': data_store['time'],
		    'source': data_store['source']
		})
		
		# Reset the data store for the next set of 50 questions
		data_store['question'].clear()
		data_store['answer'].clear()
		data_store['context'].clear()
		data_store['time'].clear()
		data_store['source'].clear()
		#data_store['count'] = 0
		
		# Save the DataFrame to a CSV file
		csv_filename = f"CSV_Result_med/{db}{model}_{file_name}"
		df.to_csv(csv_filename, index=False)


		# Print success message
		print(f"Data saved as '{csv_filename}'.")
		

# Wrap text to a specified width while preserving newlines
def wrap_text_preserve_newlines(text, width=110):
	lines = text.split('\n')
	wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
	wrapped_text = '\n'.join(wrapped_lines)
	return wrapped_text

# Process the response and print sources
def process_llm_response(llm_response):
	print('\n\nSources:')
	for source in llm_response["source_documents"]:
		print(source.metadata['source'])
        
# Select the appropriate prompt template based on the model name
def process_prompt(args):
	if args.model_name[0]=='Q':
		return qwen_rag_prompt_template
	if args.model_name[0]=='P':
		return phi_rag_prompt_template


