from imports import * 
from utils import *
from config import *

# Get process ID
pid = os.getpid()

# Parse the model name input from command line
args = parse_args()

qwen_rag_prompt_template = process_prompt(args)

model_id=inizialize(args)

# Load the existing vector database instead of creating a new one
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY_DB, 
		  embedding_function=HuggingFaceBgeEmbeddings(model_name=DB_EMBD_MODEL_NAME, model_kwargs={"device": DEVICE_MAP}))

retriever = vectordb.as_retriever(search_type=SEARCH_TYPE_RETRIEVER, search_kwargs={"k": RETRIVED_K, "score_threshold": SIMILARITY_THRESHOLD})

model = HuggingFaceCrossEncoder(model_name=RERANKER_MODEL)
compressor = CrossEncoderReranker(model=model, top_n=RETRIEVER_TOP_N)
compression_retriever = ContextualCompressionRetriever(
	base_compressor=compressor, base_retriever=retriever
)

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False, clean_up_tokenization_spaces=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map=DEVICE_MAP)

pipe = pipeline(
          MODEL_REPLY,
          model=model,
          tokenizer=tokenizer,
          max_new_tokens=MAX_NEW_TOKENS,
          clean_up_tokenization_spaces=True,
          device_map=DEVICE_MAP
)

local_llm = HuggingFacePipeline(pipeline=pipe)

# Create the retrieval chain
combine_docs_chain = create_stuff_documents_chain(local_llm, PromptTemplate(input_variables=["context", "question"], template=qwen_rag_prompt_template))

# Create the full RAG chain with the compression retriever and the combine_docs_chain
rag_chain = create_retrieval_chain(compression_retriever, combine_docs_chain)

# Ask question in input when question file is not given
if args.test is None:
	while True:
		q=input("How can I help you?\n>>")
		if q=="exit":
			break
		start = time.time()
		llm_response = rag_chain.invoke(input={"question": q, "input": q})
		
		print("question:\n", llm_response['question'])
		print("answer:\n", split_answer(llm_response['answer']))
		print("--------------------------------------------------------------------")


### ------------------------------------------------------
### WHEN GIVEN A CSV FILE WITH QUESTIONS
else:
	data = pd.read_csv(args.test)
	
	for n in range(0, len(data)):
		q=data['Question'][n]
		g=data['ground_truths'][n]
		print("Question: \n", data['Question'][n])
		start = time.time()
		llm_response = rag_chain.invoke(input={"question": q, "input": q})

		store_answer(db=PERSIST_DIRECTORY_DB, model=args.model_name, file_name=args.test, llm_response=llm_response, time=(time.time() - start), ground_truth_data=g, save_file=(True if n==(len(data)-1) else False))
		print("\nAnswer:\n", split_answer(llm_response['answer']))
		print("--------------------------------------------------------------------")

