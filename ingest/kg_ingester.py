"""
Knowledge Graph Builder

This module contains the KnowledgeGraphBuilder class, which manages the knowledge graph.
It includes methods for processing PDFs, adding nodes and edges, deleting nodes and edges,
and exporting the graph for visualization.
base: claude.ai/chat/c72b9301-8e70-47b3-ae9f-0a801b621b93
"""

import os
import re
import pickle
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
import networkx as nx
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.schema import Document
from dotenv import load_dotenv
from loguru import logger
import langchain.docstore.document as docstore
import tiktoken
import time
# local imports
from ingest.file_parser import FileParser
from query.llm_creator import LLMCreator
from ingest.splitter_creator import SplitterCreator
import settings
import utils as ut


class KGIngester:
    def __init__(self,
                #  collection_name: str,
                 content_folder: str,
                 kg_path: str,
                 document_selection: List[str] = None, # type: ignore
                 llm_provider: str = None, # type: ignore
                 llm_model: str = None, # type: ignore
                 embeddings_provider: str = None, # type: ignore
                 embeddings_model: str = None, # type: ignore
                #  retriever_type: str = None,
                 text_splitter_method: str = None, # type: ignore
                 chunk_size: int = None, # type: ignore
                 chunk_overlap: int = None) -> None: # type: ignore
        
        load_dotenv(dotenv_path=os.path.join(settings.ENVLOC, ".env"))
        # self.collection_name = collection_name
        self.content_folder = content_folder
        self.kg_path = kg_path
        self.document_selection = document_selection
        self.llm_provider = settings.LLM_PROVIDER if llm_provider is None else llm_provider
        self.llm_model = settings.LLM_MODEL if llm_model is None else llm_model
        self.embeddings_provider = settings.EMBEDDINGS_PROVIDER if embeddings_provider is None else embeddings_provider
        self.embeddings_model = settings.EMBEDDINGS_MODEL if embeddings_model is None else embeddings_model
        # self.retriever_type = settings.RETRIEVER_TYPE if retriever_type is None else retriever_type
        self.text_splitter_method = settings.TEXT_SPLITTER_METHOD \
            if text_splitter_method is None else text_splitter_method
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

        graph_name = "knowledge_graph.pkl"
        self.graph_path = os.path.join(self.kg_path, graph_name)
        # load (when existing) or create a knowledge graph object
        self.graph = self._load_or_create_graph()
        
        # define llm
        self.llm = LLMCreator(self.llm_provider,
                              self.llm_model).get_llm()

        self.graph_transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=["Person", "Organization", "Location", "Event", 
                          "Concept", "Technology", "Date", "Product"],
            allowed_relationships=["RELATED_TO", "LOCATED_IN", "WORKS_FOR", 
                                 "CREATED", "PARTICIPATED_IN", "MENTIONS",
                                 "OCCURRED_ON", "USES", "PRODUCES"],
            node_properties=["description", "source_page", "source_file"],
            relationship_properties=["description", "source"],
            strict_mode=False
        )

        self.text_splitter = SplitterCreator(self.text_splitter_method,
                                             self.chunk_size,
                                             self.chunk_overlap).get_splitter()

        self.processed_files = self._load_processed_files()
        
    def merge_hyphenated_words(self, text: str) -> str:
        """
        Merge words in the text that have been split with a hyphen.
        """
        return re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    def fix_newlines(self, text: str) -> str:
        """
        Replace single newline characters in the text with spaces.
        """
        return re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    def remove_multiple_newlines(self, text: str) -> str:
        """
        Reduce multiple newline characters in the text to a single newline.
        """
        return re.sub(r"\n{2,}", "\n", text)

    def clean_texts(self,
                    texts: List[Tuple[int, str]],
                    cleaning_functions: List[Callable[[str], str]]
                    ) -> List[Tuple[int, str]]:
        """
        Apply the cleaning functions to the text of each page.
        """
        logger.info("Cleaning texts")
        cleaned_texts = []
        for page_num, text in texts:
            for cleaning_function in cleaning_functions:
                text = cleaning_function(text)
            cleaned_texts.append((page_num, text))

        return cleaned_texts

    def texts_to_docs(self,
                      texts: List[Tuple[int, str]],
                      metadata: Dict[str, str]) -> List[docstore.Document]:
        """
        Split the text into chunks and return them as Documents.
        """
        docs: List[docstore.Document] = []
        splitter_language = ut.LANGUAGE_MAP.get(metadata['Language'], 'english')
        splitter = SplitterCreator(self.text_splitter_method,
                                   self.chunk_size,
                                   self.chunk_overlap).get_splitter(splitter_language)
        prv_page_num = -1
        for page_num, text in texts:
            logger.info(f"Splitting text from page {page_num}")
            # reset chunk number to 0 only when text is from new page
            if page_num != prv_page_num:
                chunk_num = 0
            chunk_texts = splitter.split_text(text)
            # !! chunk_texts can contain duplicates (experienced with ingestion of .txt files)
            # Deduplicate the list chunk_texts
            chunk_texts = list(dict.fromkeys(chunk_texts))
            for chunk_text in chunk_texts:
                metadata_combined = {
                    "page_number": page_num,
                    "chunk": chunk_num,
                    "source": f"p{page_num}-{chunk_num}",
                    **metadata,
                }
                doc = docstore.Document(
                    page_content=chunk_text,
                    # metadata_combined = {"title": , "author": , "indicator_url": , "indicator_closed": ,
                    #                      "filename": , "Language": , "last_change_time": , "page_number": ,
                    #                      "chunk": , "source": }
                    metadata=metadata_combined
                )
                docs.append(doc)
                chunk_num += 1
                prv_page_num = page_num

        return docs

    def clean_texts_to_docs(self, raw_texts, metadata) -> List[docstore.Document]:
        """"
        Combines the functions clean_text and text_to_docs
        """
        cleaning_functions: List = [
            self.merge_hyphenated_words,
            self.fix_newlines,
            self.remove_multiple_newlines
        ]
        cleaned_texts = self.clean_texts(raw_texts, cleaning_functions)
        docs = self.texts_to_docs(texts=cleaned_texts,
                                  metadata=metadata)

        return docs

    def count_ada_tokens(self, raw_texts: List[Tuple[int, str]]) -> int:
        """
        Counts the number of tokens in the given text for OpenAI's Ada embedding model.

        Parameters:
            text (str): The input text.

        Returns:
            int: The number of tokens.
        """
        total_tokens = 0
        for _, text in raw_texts:
            # Load the tokenizer for the Ada model
            tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")
            # Encode the text and count tokens
            tokens = tokenizer.encode(text)
            total_tokens += len(tokens)
        return total_tokens


    # def _load_or_create_graph(self) -> nx.MultiDiGraph:
    #     if os.path.exists(self.graph_path):
    #         with open(self.graph_path, 'rb') as f:
    #             return pickle.load(f)
    #     return nx.MultiDiGraph()
    
    def _load_or_create_graph(self) -> nx.DiGraph:
        if os.path.exists(self.graph_path):
            with open(self.graph_path, 'rb') as f:
                return pickle.load(f)
        return nx.DiGraph()
    
    def _save_graph(self):
        with open(self.graph_path, 'wb') as f:
            pickle.dump(self.graph, f)
    
    def _load_processed_files(self) -> set:
        processed_file_path = self.graph_path.replace('.pkl', '_processed.json')
        if os.path.exists(processed_file_path):
            with open(processed_file_path, 'r') as f:
                return set(json.load(f))
        return set()
    
    def _save_processed_files(self):
        processed_file_path = self.graph_path.replace('.pkl', '_processed.json')
        print(f"processed_file_path = {processed_file_path}")
        with open(processed_file_path, 'w') as f:
            json.dump(list(self.processed_files), f)
    
    def _add_graph_document_to_networkx(self, graph_doc: Any, source_file: str):
        for node in graph_doc.nodes:
            node_id = f"{node.type}:{node.id}"
            print(f"node.id = {node.id}")
            print(f"node.type = {node.type}")
            print(f"node.properties = {node.properties}")
            print(f"node = {node}")
            
            if node_id not in self.graph:
                self.graph.add_node(
                    node_id,
                    label=node.id,
                    type=node.type,
                    properties=node.properties if hasattr(node, 'properties') else {},
                    source_file=source_file
                )
        
        for rel in graph_doc.relationships:
            source_id = f"{rel.source.type}:{rel.source.id}"
            target_id = f"{rel.target.type}:{rel.target.id}"
            self.graph.add_edge(
                source_id,
                target_id,
                type=rel.type,
                properties=rel.properties if hasattr(rel, 'properties') else {},
                source_file=source_file
            )
    
    def add_node(self, node_id: str, node_type: str, properties: Dict = None) -> bool:
        full_id = f"{node_type}:{node_id}"
        
        if full_id in self.graph:
            return False
        
        self.graph.add_node(
            full_id,
            label=node_id,
            type=node_type,
            properties=properties or {},
            source_file="manual_entry"
        )
        
        self._save_graph()
        return True
    
    def add_edge(self, source_id: str, target_id: str, 
                 relationship_type: str, properties: Dict = None) -> bool:
        if source_id not in self.graph or target_id not in self.graph:
            return False
        
        self.graph.add_edge(
            source_id,
            target_id,
            type=relationship_type,
            properties=properties or {},
            source_file="manual_entry"
        )
        
        self._save_graph()
        return True
    
    def delete_node(self, node_id: str) -> bool:
        if node_id not in self.graph:
            return False
        
        self.graph.remove_node(node_id)
        self._save_graph()
        return True
    
    def delete_edge(self, source_id: str, target_id: str, key: int = 0) -> bool:
        if self.graph.has_edge(source_id, target_id, key=key):
            self.graph.remove_edge(source_id, target_id, key=key)
            self._save_graph()
            return True
        return False
    
    def get_graph_stats(self) -> Dict:
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "node_types": self._count_node_types(),
            "edge_types": self._count_edge_types(),
            "processed_files": list(self.processed_files)
        }
    
    def _count_node_types(self) -> Dict[str, int]:
        type_counts = {}
        for node_id, data in self.graph.nodes(data=True):
            node_type = data.get('type', 'Unknown')
            type_counts[node_type] = type_counts.get(node_type, 0) + 1
        return type_counts
    
    def _count_edge_types(self) -> Dict[str, int]:
        type_counts = {}
        for _, _, data in self.graph.edges(data=True):
            edge_type = data.get('type', 'Unknown')
            type_counts[edge_type] = type_counts.get(edge_type, 0) + 1
        return type_counts
    
    def export_to_pyvis(self, output_path: str = "knowledge_graph.html"):
        pass  # Implementation for exporting to PyVis can be added later


    def build(self) -> None:
        """
        Ingests all relevant files in the folder
        Checks are done whether vector store needs to be synchronized with folder contents

        REPLACES process_pdf()

        """
        # create empty list representing added files
        files_added = []

        # get all relevant files in the folder
        relevant_files_in_folder_selected = ut.get_relevant_files_in_folder(self.content_folder,
                                                                            self.document_selection)
        # relevant_files_in_folder_all = ut.get_relevant_files_in_folder(self.content_folder)
        # if the vector store already exists, get the set of ingested files from the vector store
        # if os.path.exists(self.kg_folder):
        #     # get knowledge graph
        #     vector_store = VectorStoreCreator().get_vectorstore(embeddings,
        #                                                         self.collection_name,
        #                                                         self.vecdb_folder)
        #     logger.info(f"Vector store already exists for specified settings and folder {self.content_folder}")
        #     # determine the files that are added or deleted
        #     collection = vector_store.get()  # dict_keys(['ids', 'embeddings', 'documents', 'metadatas'])
            # files_in_kg = [metadata['filename'] for metadata in collection['metadatas']]
            # files_in_store = list(set(files_in_store))

            # check if files were added, removed or updated
            # files_added depend on the selection that has been made
            # files_updated depend on the selction that has been made
            # files_deleted depend on all files in the folder
        files_added = [file for file in relevant_files_in_folder_selected if file not in self.processed_files]
        # print(files_added)
        #     files_deleted = [file for file in files_in_store if file not in relevant_files_in_folder_all]
        #     # Check for last changed date
        #     filename_lastchange_dict = {metadata['filename']: metadata.get('last_change_time', None)
        #                                 for metadata in collection['metadatas']}
        #     files_updated = [file for file in relevant_files_in_folder_selected
        #                      if (file not in files_added) and
        #                         (filename_lastchange_dict[file] !=
        #                          os.stat(os.path.join(self.content_folder, file)).st_mtime)]

        #     # delete from vector store all chunks associated with deleted or updated files
        #     to_delete = files_deleted + files_updated
        #     if len(to_delete) > 0:
        #         logger.info(f"Files are deleted, so vector store for {self.content_folder} needs to be updated")
        #         idx_id_to_delete = []
        #         for idx in range(len(collection['ids'])):
        #             idx_id = collection['ids'][idx]
        #             idx_metadata = collection['metadatas'][idx]
        #             if idx_metadata['filename'] in to_delete:
        #                 idx_id_to_delete.append(idx_id)
        #                 if idx_metadata['filename'].endswith(".docx"):
        #                     os.remove(os.path.join(self.content_folder,
        #                                            "conversions",
        #                                            idx_metadata['filename'] + ".pdf"))
        #         vector_store.delete(idx_id_to_delete)
        #         logger.info("Deleted files from vectorstore")

        #     # add to vector store all chunks associated with added or updated files
        #     to_add = files_added + files_updated
        # to_add = files_added
        # else it needs to be created first
        # else:
        #     logger.info(f"Vector store to be created for folder {self.content_folder}")
        #     # get chroma vector store
        #     vector_store = VectorStoreCreator().get_vectorstore(embeddings,
        #                                                         self.collection_name,
        #                                                         self.vecdb_folder)
        #     # all relevant files in the folder are to be ingested into the vector store
        # to_add = list(relevant_files_in_folder_selected)

        # # If there are any files to be ingested into the vector store
        if len(files_added) > 0:
            logger.info(f"Files are added, so knowledge graph for {self.content_folder} needs to be updated")
            # create FileParser object
            file_parser = FileParser()

            initial_nodes = self.graph.number_of_nodes()
            initial_edges = self.graph.number_of_edges()
            
            for file in files_added:
                file_path = os.path.join(self.content_folder, file)
                # extract raw text pages and metadata according to file type
                logger.info(f"Parsing file {file}")
                raw_texts, metadata = file_parser.parse_file(file_path)
                # count tokens
                tokens_document = self.count_ada_tokens(raw_texts)
                # define documents according to size. If the amount of tokens is larger thatn the rate limit woud allow,
                # then split documents and pause in loop
                documents = self.clean_texts_to_docs(raw_texts=raw_texts,
                                                     metadata=metadata)
                logger.info(f"Extracted {len(documents)} chunks (Tokens: {tokens_document}) from {file}")
                if tokens_document > 40_000:
                    logger.info("Pause for 10 seconds to avoid hitting rate limit")
                    time.sleep(10)
                try:
                    graph_documents = self.graph_transformer.convert_to_graph_documents(documents)
                    
                    for graph_doc in graph_documents:
                        self._add_graph_document_to_networkx(graph_doc, file)
                        
                except Exception:
                    continue
                
                self.processed_files.add(file)
            self._save_processed_files()
            self._save_graph()
            
            new_nodes = self.graph.number_of_nodes() - initial_nodes
            new_edges = self.graph.number_of_edges() - initial_edges
            
        print(f"new_nodes: {new_nodes} , new_edges: {new_edges}")