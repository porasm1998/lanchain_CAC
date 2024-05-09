import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { pinecone } from '@/utils/pinecone-client';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { PINECONE_INDEX_NAME, PINECONE_NAME_SPACE } from '@/config/pinecone';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { Document } from 'langchain/document';

// Path to the directory containing PDF files
const filePath = 'docs';

// Load and Split PDFs with Error Handling
const loadAndSplitDocuments = async (): Promise<Document[]> => {
  try {
    // Load all PDF files from the directory
    const directoryLoader = new DirectoryLoader(filePath, {
      '.pdf': (path) => new PDFLoader(path),
    });

    const rawDocs = await directoryLoader.load();
    console.log('Loaded raw documents:', rawDocs.length);

    // Split the loaded documents into smaller chunks
    const textSplitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000,
      chunkOverlap: 200,
    });

    const splitDocs = await textSplitter.splitDocuments(rawDocs);
    console.log('Split documents:', splitDocs.length);

    return splitDocs;
  } catch (error) {
    console.error('Error loading and splitting documents:', error);
    return [];
  }
};

// Run the Ingestion Process
export const run = async () => {
  try {
    const docs = await loadAndSplitDocuments();

    if (docs.length === 0) {
      throw new Error('No documents to ingest');
    }

    console.log('Creating vector store...');

    // Initialize OpenAI Embeddings and Pinecone Index
    const embeddings = new OpenAIEmbeddings();
    const index = pinecone.Index(PINECONE_INDEX_NAME);

    // Create Pinecone Vector Store from Documents
    await PineconeStore.fromDocuments(docs, embeddings, {
      pineconeIndex: index,
      namespace: PINECONE_NAME_SPACE || 'default-namespace',
    });

    console.log('Ingestion complete');
  } catch (error) {
    console.error('Error during ingestion:', error);
    throw new Error('Failed to ingest your data');
  }
};

(async () => {
  await run();
})();
