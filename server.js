import express from 'express';
import { ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { createToolCallingAgent, AgentExecutor } from "langchain/agents";
import { DynamicStructuredTool } from "langchain/tools";
import { ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate } from "langchain/prompts";
import { z } from "zod";
import * as dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// 1. LLM Setup
const model = new ChatGoogleGenerativeAI({
  model: "gemini-1.5-flash",
  temperature: 0,
  apiKey: process.env.GOOGLE_API_KEY,
});

// 2. Custom Dynamic Tool: Get Menu
const getMenuTool = new DynamicStructuredTool({
  name: "get_menu_tool",
  description: "Use this tool to get the restaurant menu for a specific meal category (breakfast, lunch, or dinner). Returns the final answer for the given category.",
  schema: z.object({
    category: z.string().describe("The type of meal: 'breakfast', 'lunch', or 'dinner'."),
  }),
  func: async ({ category }) => {
    const menus = {
      breakfast: "Aloo Paratha, Poha, Masala Chai",
      lunch: "Paneer Butter Masala, Dal Fry, Jeera Rice, Roti",
      dinner: "Veg Biryani, Raita, Salad, Gulab Jamun"
    };
    
    const lowerCategory = category.toLowerCase();
    if (menus[lowerCategory]) {
      return menus[lowerCategory];
    }
    return "No menu found for that category. Please ask for breakfast, lunch, or dinner.";
  },
});

// 3. Agent & Prompt Setup
const prompt = ChatPromptTemplate.fromMessages([
  SystemMessagePromptTemplate.fromTemplate("You are a helpful restaurant assistant. Use the 'get_menu_tool' to answer questions about the menu for breakfast, lunch, or dinner."),
  HumanMessagePromptTemplate.fromTemplate("{input}"),
]);

const agent = await createToolCallingAgent({
  llm: model,
  tools: [getMenuTool],
  prompt: prompt,
});

// 4. Agent Executor with Iteration Fix (from transcript)
const executor = new AgentExecutor({
  agent,
  tools: [getMenuTool],
  verbose: true, // Enable for debugging in terminal
  maxIterations: 10,
  returnIntermediateSteps: true, // Critical for the "hack"
});

// 5. API Endpoint: /api/chat
app.post("/api/chat", async (req, res) => {
  const userInput = req.body.input;
  
  try {
    const response = await executor.invoke({
      input: userInput,
    });

    // HACK: Handle "Agent stopped due to max iterations" by using intermediate steps
    if (response.output && response.output.includes("Agent stopped due to max iterations")) {
       if (response.intermediateSteps && response.intermediateSteps.length > 0) {
          const lastStep = response.intermediateSteps[response.intermediateSteps.length - 1];
          if (lastStep.observation) {
             return res.json({ output: lastStep.observation });
          }
       }
    }

    if (response.output) {
      return res.json({ output: response.output });
    } else {
      return res.status(500).json({ output: "Sorry, the agent could not find an answer." });
    }

  } catch (error) {
    console.error("Error during agent execution:", error);
    return res.status(500).json({ output: "Sorry, something went wrong. Please try again later." });
  }
});

// Serve Frontend
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});
