"use client";
import pineconeClient from "@/lib/pinecone";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import OpenAI from "openai";
import { PineconeStore } from "langchain/vectorstores/pinecone";
import { useState } from "react";
import { load } from "langchain/load";
import { HuggingFaceInferenceEmbeddings } from "@langchain/community/embeddings/hf";
import { RunnableSequence } from "@langchain/core/runnables";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { PromptTemplate } from "@langchain/core/prompts";
import { formatDocumentsAsString } from "langchain/util/document";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { HuggingFaceInference } from "langchain/llms/hf";
const apiKey = "API_KEY";

export default function Home() {
  const [query, setQeury] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [namespace, setNamespace] = useState("quill");
  const [desc, setDesc] = useState("");

  const embeddings2 = new HuggingFaceInferenceEmbeddings({
    apiKey: "", // In Node.js defaults to process.env.HUGGINGFACEHUB_API_KEY
  });

  const handleClicked = async () => {
    try {
      setLoading(true);
      await pineconeClient.getConfig();

      console.log("pineconeClient", pineconeClient);
      //const index = pc.index('products');
      const pineconeIndex = pineconeClient.index("quill");

      pineconeClient.createIndex({
        name: "quill",
        dimension: 768,
        metric: "cosine",
        spec: {
          pod: {
            environment: "gcp-starter",
            podType: "starter",
          },
        },
      });
      console.log("desc", desc);

      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
      });

      const docs = await textSplitter.splitText(desc);
      console.log("docs", docs);

      await PineconeStore.fromTexts(docs, {}, embeddings2, {
        pineconeIndex,
        namespace: namespace,
      });

      console.log("Index created");
      setLoading(false);
    } catch (e) {
      console.log(e);
    }
  };

  const handleQuery = async () => {
    console.log("namespace", namespace);
    try {
      await pineconeClient.getConfig();
      const pineconeIndex = pineconeClient.index("quill");

      const vectorStore = await PineconeStore.fromExistingIndex(embeddings2, {
        pineconeIndex,
        namespace: namespace,
      });

      const retriever = vectorStore.asRetriever();

      const formatChatHistory = (
        human: string,
        ai: string,
        previousChatHistory?: string
      ) => {
        const newInteraction = `Human: ${human}\nAI: ${ai}`;
        if (!previousChatHistory) {
          return newInteraction;
        }
        return `${previousChatHistory}\n\n${newInteraction}`;
      };

      const questionPrompt = PromptTemplate.fromTemplate(
        `Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
        ----------------
        CONTEXT: {context}
        ----------------
        CHAT HISTORY: {chatHistory}
        ----------------
        QUESTION: {question}
        ----------------
        Helpful Answer:`
      );

      const model = new HuggingFaceInference({
        apiKey: "",
        model: "mistralai/Mistral-7B-Instruct-v0.2",
      });

      const chain = RunnableSequence.from([
        {
          question: (input: { question: string; chatHistory?: string }) =>
            input.question,
          chatHistory: (input: { question: string; chatHistory?: string }) =>
            input.chatHistory ?? "",
          context: async (input: {
            question: string;
            chatHistory?: string;
          }) => {
            const relevantDocs = await retriever.invoke(input.question);
            const serialized = formatDocumentsAsString(relevantDocs);
            return serialized;
          },
        },
        questionPrompt,
        model,
        new StringOutputParser(),
      ]);

      const results = await vectorStore.similaritySearch(query, 2);
      console.log("Results", results);

      const questionOne = "What did the president say about Justice Breyer?";

      const resultOne = await chain.invoke({
        question: query,
      });
      console.log("resultOne", resultOne);
      setResponse(JSON.stringify(results, null, 2));
    } catch (e) {
      console.log(e);
    }
  };

  return (
    <main className="flex flex-col items-center ">
      <h1 className="text-2xl font-bold mt-2">Pinecone</h1>

      <input
        className="w-11/12 mt-3 mb-3 p-4 border-2 border-gray-300 rounded-lg resize-none focus:outline-none focus:border-blue-500 mx-12"
        type="text"
        placeholder="give a namespace"
        onChange={(e) => setNamespace(e.target.value!)}
      />
      <textarea
        className="w-11/12 mt-3 h-96 p-4 border-2 border-gray-300 rounded-lg resize-none focus:outline-none focus:border-blue-500 mx-12"
        name="paragraph"
        id=""
        cols={30}
        rows={10}
        placeholder="Enter text here"
        onChange={(e) => setDesc(e.target.value!)}
      ></textarea>

      <button
        onClick={handleClicked}
        className="bg-blue-500 text-white px-4 py-2 rounded-lg focus:outline-none"
      >
        Index
      </button>

      {!loading && (
        <>
          <input
            onChange={(e) => setQeury(e.target.value)}
            type="text"
            className="w-11/12 mt-3 mb-3 p-4 border-2 border-gray-300 rounded-lg resize-none focus:outline-none focus:border-blue-500 mx-12"
            placeholder="Enter text here"
          />

          <button
            onClick={handleQuery}
            className="bg-blue-500 text-white px-4 py-2 rounded-lg focus:outline-none"
          >
            Query
          </button>
        </>
      )}
      {response && (
        <div className="w-11/12 mt-3 p-4 border-2 border-gray-300 rounded-lg resize-none focus:outline-none focus:border-blue-500 mx-12">
          {response}
        </div>
      )}
    </main>
  );
}
