"""
This module allows the querying of a networkx graph object.
It assumes that the graph object already exists, created by running kg_ingest.py.
"""
# imports
import os
from loguru import logger
# local imports
from query.kg_querier import KGQuerier
import utils as ut
import settings


def main():
    # get source folder with docs from user
    content_folder_path = input("Source folder of documents (including path): ")
    # Get content folder name from path
    content_folder_name = os.path.basename(content_folder_path)
    # Get private docs indicator from user
    # confidential_yn = input("Are there any confidential documents in the folder? (y/n) ")
    # confidential = confidential_yn in ["y", "Y"]
    confidential = False
    # get relevant models
    # llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
    #                                                                                         private=confidential)

    kg_folder_path = ut.create_kg_folder_path(content_folder_path=content_folder_path)

    # obtain the networkx graph object
    graph = ut.get_graph(os.path.join(kg_folder_path,"knowledge_graph.pkl"))

    # create instance of Querier once
    kg_querier = KGQuerier(
        graph=graph
    )

    # if vector store folder does not exist, stop
    if not os.path.exists(kg_folder_path):
        logger.info("There is no knowledge graph for this folder yet. First run \"python kg_ingest.py\"")
        ut.exit_program()
    else:
        # else create the query chain
        kg_querier.make_chain(content_folder_name, kg_folder_path)
        while True:
            # get question from user
            question = input("Question: ")
            if question not in ["exit", "quit", "q"]:
                # generate answer and include sources used to produce that answer
                response = kg_querier.ask_question(question)
                # logger.info(f"\nAnswer: {response['answer']}")
                logger.info(f"\nAnswer: {response}")
                # # if the retriever returns one or more chunks with a score above the threshold
                # if len(response["source_documents"]) > 0:
                #     # log the answer to the question and the sources used for creating the answer
                #     logger.info("\nSources:\n")
                #     for document in response["source_documents"]:
                #         logger.info(f"File {document.metadata['filename']}, \
                #                     Page {document.metadata['page_number'] + 1}, \
                #                     chunk text: {document.page_content}\n")
            else:
                ut.exit_program()


if __name__ == "__main__":
    main()
