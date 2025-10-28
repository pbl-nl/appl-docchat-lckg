"""
This module can be run in stand alone fashion using 'python ingest.py'.
It ingests documents when necessary from a given folder into a persistent vector database
"""
import os
from loguru import logger
# local imports
from ingest.kg_ingester import KGIngester
import utils as ut


def main():
    """
    Ingests documents when necessary from a given folder into a persistent vector database
    """
    # Get source folder with docs from user
    content_folder_path = input("Source folder path of documents (including path): ")
    
    # Get content folder name from path
    content_folder_name = os.path.basename(content_folder_path)
    
    # Get private docs indicator from user
    # confidential_yn = input("Are there any confidential documents in the folder? (y/n) ")
    # confidential = confidential_yn in ["y", "Y"]
    confidential = False
    
    # get relevant models
    llm_provider, llm_model, embeddings_provider, embeddings_model = ut.get_relevant_models(summary=False,
                                                                                            private=confidential)
    # # create folder name for knowledge graph according to settings
    # kg_folder_name = ut.create_kg_folder_name(content_folder_path=content_folder_path,
    #                                           llm_provider=llm_provider,
    #                                           llm_model=llm_model,
    #                                           embeddings_provider=embeddings_provider,
    #                                           embeddings_model=embeddings_model)

    # # create subfolder for storage of knowledge graph if not existing
    # ut.create_kg_folder(content_folder_path, kg_folder_name)
    # kg_folder_path = os.path.join(content_folder_path, "knowledge_graphs", kg_folder_name)

    kg_folder_path = ut.create_kg_folder_path(content_folder_path=content_folder_path)
    
    # create kgbuilder object
    kgbuilder = KGIngester(content_folder=content_folder_path,
                           kg_path=kg_folder_path,
                           llm_provider=llm_provider,
                           llm_model=llm_model,
                           embeddings_provider=embeddings_provider,
                           embeddings_model=embeddings_model)
    # build or extend knowledge graph
    kgbuilder.build()
    logger.info(f"finished building knowledge graph for folder {content_folder_path}")


if __name__ == "__main__":
    main()
